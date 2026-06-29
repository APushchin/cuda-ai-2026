import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

def layernorm_pycuda(input, gamma, beta, row_size, eps=1e-5):
    """
    Apply Layer Normalization to each row of the input matrix.
    
    Parameters
    ----------
    input : list or numpy.ndarray of float
        Flattened matrix in row‑major order. Its length must be divisible by row_size.
    gamma : list or numpy.ndarray of float
        Scale parameter, length = row_size.
    beta : list or numpy.ndarray of float
        Shift parameter, length = row_size.
    row_size : int
        Number of features per row (i.e., number of columns).
    eps : float, optional
        Small constant for numerical stability.
    
    Returns
    -------
    numpy.ndarray
        Flattened matrix of the same shape as input, containing the row‑wise
        normalized results.
    """
    # Convert inputs to numpy arrays with float32 dtype
    input = np.array(input, dtype=np.float32).ravel()
    gamma = np.array(gamma, dtype=np.float32).ravel()
    beta = np.array(beta, dtype=np.float32).ravel()
    
    # Calculate number of rows and total elements
    total_elements = input.size
    num_rows = total_elements // row_size
    col_size = row_size  # Number of columns = features per row
    
    # Validate input
    if total_elements != num_rows * row_size:
        raise ValueError(f"Input size {total_elements} is not divisible by row_size {row_size}")
    
    if gamma.size != row_size or beta.size != row_size:
        raise ValueError(f"gamma and beta must have length {row_size}")
    
    # Constants for kernel configuration
    BLOCK_SIZE = 256
    WARP_SIZE = 32
    NUM_WARPS = BLOCK_SIZE // WARP_SIZE
    
    # CUDA kernel source code
    kernel_source = f"""
    #define BLOCK_SIZE {BLOCK_SIZE}
    #define WARP_SIZE {WARP_SIZE}
    #define NUM_WARPS {NUM_WARPS}
    
    __device__ __inline__ float warpReduceSum(float val) {{
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {{
            val += __shfl_down_sync(0xffffffff, val, offset);
        }}
        return val;
    }}
    
    __device__ __inline__ float blockReduceSum(float val, float* shared_cache) {{
        int tid = threadIdx.x;
        int lane = tid % WARP_SIZE;
        int wid = tid / WARP_SIZE;
        
        // Warp-level reduction
        val = warpReduceSum(val);
        
        // First warp lane stores warp sums in shared memory
        if (lane == 0) shared_cache[wid] = val;
        __syncthreads();
        
        // Reduce warp sums (only first warp participates)
        float final_sum = (tid < NUM_WARPS) ? shared_cache[lane] : 0.0f;
        if (wid == 0) final_sum = warpReduceSum(final_sum);
        
        // Broadcast result to all threads
        if (tid == 0) shared_cache[0] = final_sum;
        __syncthreads();
        
        return shared_cache[0];
    }}
    
    __global__ void layernormKernel(const float* __restrict__ input,
                                    const float* __restrict__ gamma,
                                    const float* __restrict__ beta,
                                    float* __restrict__ output,
                                    int num_rows,
                                    int col_size,
                                    float eps) {{
        int row = blockIdx.x;
        if (row >= num_rows) return;
        
        int tid = threadIdx.x;
        __shared__ float shared_cache[NUM_WARPS];
        
        // Get pointer to current row
        const float* row_input = input + row * col_size;
        float* row_output = output + row * col_size;
        
        // ===== Phase 1: Compute mean =====
        float local_sum = 0.0f;
        for (int col = tid; col < col_size; col += BLOCK_SIZE) {{
            local_sum += row_input[col];
        }}
        float row_mean = blockReduceSum(local_sum, shared_cache) / col_size;
        
        // ===== Phase 2: Compute variance =====
        float local_var_sum = 0.0f;
        for (int col = tid; col < col_size; col += BLOCK_SIZE) {{
            float diff = row_input[col] - row_mean;
            local_var_sum += diff * diff;
        }}
        float row_variance = blockReduceSum(local_var_sum, shared_cache) / col_size;
        float inv_std = rsqrtf(row_variance + eps);
        
        // ===== Phase 3: Apply normalization, scale, and shift =====
        for (int col = tid; col < col_size; col += BLOCK_SIZE) {{
            float normalized = (row_input[col] - row_mean) * inv_std;
            row_output[col] = gamma[col] * normalized + beta[col];
        }}
    }}
    """
    
    # Compile the kernel
    mod = SourceModule(kernel_source)
    layernorm_kernel = mod.get_function("layernormKernel")
    
    # Allocate device memory
    input_gpu = cuda.mem_alloc(input.nbytes)
    gamma_gpu = cuda.mem_alloc(gamma.nbytes)
    beta_gpu = cuda.mem_alloc(beta.nbytes)
    output_gpu = cuda.mem_alloc(input.nbytes)
    
    # Copy data to device
    cuda.memcpy_htod(input_gpu, input)
    cuda.memcpy_htod(gamma_gpu, gamma)
    cuda.memcpy_htod(beta_gpu, beta)
    
    # Configure and launch kernel
    grid_size = (num_rows + BLOCK_SIZE - 1) // BLOCK_SIZE
    layernorm_kernel(
        input_gpu, gamma_gpu, beta_gpu, output_gpu,
        np.int32(num_rows), np.int32(col_size), np.float32(eps),
        block=(BLOCK_SIZE, 1, 1),
        grid=(num_rows, 1, 1)
    )
    
    # Copy results back to host
    output = np.empty_like(input)
    cuda.memcpy_dtoh(output, output_gpu)
    
    # Free device memory
    input_gpu.free()
    gamma_gpu.free()
    beta_gpu.free()
    output_gpu.free()
    
    return output