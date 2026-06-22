import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import time

mod = SourceModule("""
__global__ void layernorm_kernel(float *input, float *output, float *gamma, float *beta, int row_size, float eps) {

    int row_idx = blockIdx.x;
    int tid = threadIdx.x;
    int block_dim = blockDim.x;

    float *row_in = input + row_idx * row_size;
    float *row_out = output + row_idx * row_size;

    extern __shared__ float sdata[];

    // Mean calcualtion
    float local_sum = 0.0f;
    for (int i = tid; i < row_size; i += block_dim) {
        local_sum += row_in[i];
    }
    sdata[tid] = local_sum;
    __syncthreads();

    for (int s = block_dim / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    float mean = sdata[0] / row_size;
    __syncthreads();

    // Variance calcualtion
    float local_var_sum = 0.0f;
    for (int i = tid; i < row_size; i += block_dim) {
        float diff = row_in[i] - mean;
        local_var_sum += diff * diff;
    }
    sdata[tid] = local_var_sum;
    __syncthreads();

    for (int s = block_dim / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    float variance = sdata[0] / row_size;
    float inv_std = rsqrtf(variance + eps);

    // Normalization
    for (int i = tid; i < row_size; i += block_dim) {
        row_out[i] = ((row_in[i] - mean) * inv_std) * gamma[i] + beta[i];
    }
}
""")

def layernorm_pycuda(input_data, gamma, beta, row_size, eps=1e-5):
    """
    Apply Layer Normalization to each row of the input matrix.

    Parameters
    ----------
    input : list or numpy.ndarray of float
        Flattened matrix in row-major order. Its length must be divisible by row_size.
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
        Flattened matrix of the same shape as input, containing the row-wise
        normalized results.
    """
    # Matching data types
    input_arr = np.ascontiguousarray(input_data, dtype=np.float32)
    gamma_arr = np.ascontiguousarray(gamma, dtype=np.float32)
    beta_arr = np.ascontiguousarray(beta, dtype=np.float32)
    
    total_elements = input_arr.size
    num_rows = total_elements // row_size
    output_arr = np.empty_like(input_arr)

    # Dev mem alloc
    input_gpu = cuda.mem_alloc(input_arr.nbytes)
    output_gpu = cuda.mem_alloc(output_arr.nbytes)
    gamma_gpu = cuda.mem_alloc(gamma_arr.nbytes)
    beta_gpu = cuda.mem_alloc(beta_arr.nbytes)

    # Copy from host 2 dev
    stream = cuda.Stream()

    cuda.memcpy_htod_async(input_gpu, input_arr, stream)
    cuda.memcpy_htod_async(gamma_gpu, gamma_arr, stream)
    cuda.memcpy_htod_async(beta_gpu, beta_arr, stream)

    threads_per_block = min(256, row_size) 
    grid_size = (num_rows, 1, 1)
    shared_mem_size = threads_per_block * 4

    # Get kernel function
    func = mod.get_function("layernorm_kernel")
    
    # Launch kernel
    func(
        input_gpu, output_gpu, gamma_gpu, beta_gpu, 
        np.int32(row_size), np.float32(eps),
        block=(threads_per_block, 1, 1), 
        grid=grid_size, 
        shared=shared_mem_size
    )

    # Copy from dev 2 host
    cuda.memcpy_dtoh_async(output_arr, output_gpu, stream)
    stream.synchronize()
    
    # Free memory
    input_gpu.free()
    output_gpu.free()
    gamma_gpu.free()
    beta_gpu.free()

    return output_arr