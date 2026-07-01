import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

from pycuda.compiler import SourceModule

layernormKernel = r"""
#define BLOCK_SIZE 32

__global__ void LayernormKernel(const float* input, const float* gamma, const float* beta,
                           float* output, int row_size, float eps) {
    int bid = blockIdx.x;
    int tid = threadIdx.x;

    __shared__ float loc_sums[BLOCK_SIZE];
    float loc_sum = 0.f;
    for (int i = tid; i < row_size; i += BLOCK_SIZE) {
        loc_sum += input[i + bid * row_size];
    }
    loc_sums[tid] = loc_sum;
    __syncthreads();

    __shared__ float row_sum;
    __shared__ float mean;
    if (tid == 0) {
        row_sum = 0.f;
        for (int i = 0; i < BLOCK_SIZE; ++i) {
            row_sum += loc_sums[i];
        }
        mean = row_sum / row_size;
    }
    __syncthreads();

    loc_sum = 0.f;
    float x;
    for (int i = 0; i < row_size; i += BLOCK_SIZE) {
        x = input[i + bid * row_size];
        x = x - mean;
        loc_sum += x * x;
    }
    loc_sums[tid] = loc_sum;
    __syncthreads();

    __shared__ float var;
    if (tid == 0) {
        row_sum = 0.f;
        for (int i = 0; i < BLOCK_SIZE; ++i) {
            row_sum += loc_sums[i];
        }
        var = row_sum / row_size;
    }
    __syncthreads();

    for (int i = tid; i < row_size; i += BLOCK_SIZE) {
        int idx = i + bid * row_size;
        output[idx] = gamma[i] * ((input[idx] - mean) / sqrtf(var + eps)) + beta[idx];
    }
}
"""

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

    input_np = np.asarray(input, dtype=np.float32)
    gamma_np = np.asarray(gamma, dtype=np.float32)
    beta_np = np.asarray(beta, dtype=np.float32)
    output = np.zeros_like(input_np)

    input_dev = cuda.mem_alloc(input_np.nbytes)
    gamma_dev = cuda.mem_alloc(gamma_np.nbytes)
    beta_dev = cuda.mem_alloc(beta_np.nbytes)
    output_dev = cuda.mem_alloc(output.nbytes)

    cuda.memcpy_htod(input_dev, input_np)
    cuda.memcpy_htod(gamma_dev, gamma_np)
    cuda.memcpy_htod(d_beta, beta_np)

    mod = SourceModule(layernormKernel,  options=["-O3", "-use_fast_math"])
    kernel = mod.get_function("LayernormKernel")

    blk_size = 32
    row_count = input_np.size // row_size  
    kernel(input_dev, gamma_dev, d_beta, output_dev, np.int32(row_size), np.float32(eps), block=(blk_size, 1, 1), grid = (row_count, 1))

    cuda.memcpy_dtoh(output, output_dev)

    input_dev.free()
    gamma_dev.free()
    d_beta.free()
    output_dev.free()

    return output


