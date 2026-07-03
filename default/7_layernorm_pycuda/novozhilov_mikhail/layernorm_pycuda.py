import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

KERNEL_CODE = r"""
#include <math.h>
#define BLOCK_SIZE 256

__global__ void LayernormCUDA(
    float* input, 
    float* output, 
    float* gamma, 
    float* beta, 
    int row_size, 
    float eps)
{
    __shared__ float buff[BLOCK_SIZE];

    int row = blockIdx.x;
    int tx = threadIdx.x;

    float sum = 0.0f;
    for (int i = tx; i < row_size; i += BLOCK_SIZE) {
        sum += input[row * row_size + i];
    }
    buff[tx] = sum;

    __syncthreads();

    #pragma unroll
    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tx < s) {
            buff[tx] += buff[tx + s];
        }

        __syncthreads();

    }
    float mean = buff[0] / row_size;

    __syncthreads();

    float var_sum = 0.0f;
    for (int i = tx; i < row_size; i += BLOCK_SIZE) {
        float diff = input[row * row_size + i] - mean;
        var_sum += diff * diff;
    }
    buff[tx] = var_sum;

    __syncthreads();

    #pragma unroll
    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tx < s) {
            buff[tx] += buff[tx + s];
        }

        __syncthreads();

    }

    for (int i = tx; i < row_size; i += BLOCK_SIZE) {
        int idx = row * row_size + i;
        output[idx] = gamma[i] * ((input[idx] - mean) * rsqrtf((buff[0] / row_size) + eps)) + beta[i];
    }
}
"""

mod = SourceModule(KERNEL_CODE)
LayernormCUDA = mod.get_function("LayernormCUDA")

def layernorm_pycuda(input, gamma, beta, row_size, eps=1e-5):
    input = np.array(input, dtype=np.float32)
    gamma = np.array(gamma, dtype=np.float32)
    beta = np.array(beta, dtype=np.float32)
    
    size = input.size
    rows = size // row_size
    
    output = np.empty_like(input)
    
    cuda_input = cuda.mem_alloc(input.nbytes)
    cuda_output = cuda.mem_alloc(output.nbytes)
    cuda_gamma = cuda.mem_alloc(gamma.nbytes)
    cuda_beta = cuda.mem_alloc(beta.nbytes)
    
    cuda.memcpy_htod(cuda_input, input)
    cuda.memcpy_htod(cuda_gamma, gamma)
    cuda.memcpy_htod(cuda_beta, beta)
    
    LayernormCUDA(
        cuda_input, 
        cuda_output, 
        cuda_gamma, 
        cuda_beta,
        np.int32(row_size),
        np.float32(eps),
        block=(256, 1, 1),
        grid=(rows, 1, 1)
    )
    
    cuda.memcpy_dtoh(output, cuda_output)
    
    return output