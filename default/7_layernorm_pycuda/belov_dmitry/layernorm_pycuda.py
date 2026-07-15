import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np

kernel_code = """
__global__ void layerNormFunc(float *output, float *input, float *gamma, float *beta, int row_size, int col_size, int num_mtx_el, float eps)
{
    extern __shared__ float shared_mem[]; 

    int row_index = blockIdx.x; 
    int thread_index = threadIdx.x;

    float thread_sum = 0.0f;
    for (int i_el = thread_index; i_el < row_size; i_el += blockDim.x) 
    {
        thread_sum += input[row_index * row_size + i_el];
    }
    shared_mem[thread_index] = thread_sum;
    __syncthreads();
    
    __shared__ float row_mean;
    if (thread_index == 0) 
    {   
        row_mean = 0;
        for (int i_thread = 0; i_thread < blockDim.x; ++i_thread)
            row_mean += shared_mem[i_thread];

        row_mean /= row_size;
    }
    __syncthreads();

    float thread_var_sum = 0.0f;
    for (int i_el = thread_index; i_el < row_size; i_el += blockDim.x) 
    {
        float diff = input[row_index * row_size + i_el] - row_mean;
        thread_var_sum += diff * diff;
    }
    shared_mem[thread_index] = thread_var_sum;
    __syncthreads();

    __shared__ float row_inv_std;
    if (thread_index == 0) 
    {
        float variance = 0;
        for (int i_thread = 0; i_thread < blockDim.x; ++i_thread)
            variance += shared_mem[i_thread];
        variance /= row_size;

        row_inv_std = rsqrtf(variance + eps);
    }
    __syncthreads();

    for (int i = thread_index; i < row_size; i += blockDim.x) 
    {
        int global_idx = row_index * row_size + i;
        output[global_idx] = gamma[i] * (input[global_idx] - row_mean) * row_inv_std + beta[i];
    }

}
"""

mod = SourceModule(kernel_code)
layernorm_kernel = mod.get_function("layerNormFunc")

def layernorm_pycuda(input, gamma, beta, row_size, eps=1e-5):
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
    device_input = cuda.mem_alloc(input.nbytes)
    device_gamma = cuda.mem_alloc(gamma.nbytes)
    device_beta = cuda.mem_alloc(beta.nbytes)
    device_output = cuda.mem_alloc(input.nbytes)

    cuda.memcpy_htod(device_input, input)
    cuda.memcpy_htod(device_gamma, gamma)
    cuda.memcpy_htod(device_beta, beta)

    col_size = input.size / row_size

    threads_per_block = int(min(256, row_size))
    blocks_per_grid = int(col_size)
    shared_mem_size = int(threads_per_block * 4)

    layernorm_kernel(
        device_output, device_input, device_gamma, device_beta, 
        np.int32(row_size), np.int32(col_size), np.int32(input.size), np.float32(eps),
        block=(threads_per_block, 1, 1),
        grid=(blocks_per_grid, 1),
        shared=shared_mem_size
    )

    output = np.empty_like(input)
    cuda.memcpy_dtoh(output, device_output)

    device_input.free()
    device_gamma.free()
    device_beta.free()
    device_output.free()

    return output