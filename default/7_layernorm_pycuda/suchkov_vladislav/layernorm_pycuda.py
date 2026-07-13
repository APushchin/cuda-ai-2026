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

    float loc_sum = 0.0f;
    for (int col = tid; col < row_size; col += BLOCK_SIZE) {
        loc_sum += input[bid * row_size + col];
    }

    __shared__ float loc_sums[BLOCK_SIZE];
    __shared__ float row_sum;
    loc_sums[tid] = loc_sum;
    __syncthreads();

    __shared__ float mean;
    if (tid == 0) {
        row_sum = 0.0f;
        for (int i = 0; i < BLOCK_SIZE; ++i) {
            row_sum += loc_sums[i];
        }
        mean = row_sum / row_size;
    }
    __syncthreads();


    loc_sum = 0.0f;
    float x;
    for (int col = tid; col < row_size; col += BLOCK_SIZE) {
        x = input[bid * row_size + col];
        loc_sum += (x - mean) * (x - mean);
    }

    loc_sums[tid] = loc_sum;
    __syncthreads();


    __shared__ float var;
    if (tid == 0) {
        row_sum = 0.0f;
        for (int i = 0; i < BLOCK_SIZE; ++i) {
            row_sum += loc_sums[i];
        }
        var = row_sum / row_size;
    }
    __syncthreads();

    for (int col = tid; col < row_size; col += BLOCK_SIZE) {
        int idx = bid * row_size + col;
        x = (input[idx] - mean) / sqrtf(var + eps);
        output[idx] = gamma[col] * x + beta[col];
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
    cuda.memcpy_htod(beta_dev, beta_np)

    mod = SourceModule(layernormKernel,  options=["-O3", "-use_fast_math"])
    kernel = mod.get_function("LayernormKernel")

    blk_size = min(row_size, 32)
    row_count = input_np.size // row_size
    stream = cuda.Stream()
    kernel(input_dev, gamma_dev, beta_dev, output_dev, np.int32(row_size), np.float32(eps), block=(blk_size, 1, 1), grid = (row_count, 1),)

    cuda.memcpy_dtoh(output, output_dev)
    stream.synchronize()

    input_dev.free()
    gamma_dev.free()
    beta_dev.free()
    output_dev.free()

    return output
