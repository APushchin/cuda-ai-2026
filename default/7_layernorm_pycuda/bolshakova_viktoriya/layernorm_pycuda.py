import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

import numpy as np

kernel_code = """
__global__ void layer_norm_kernel(const float *__restrict__ input,
                                  const float *__restrict__ gamma,
                                  const float *__restrict__ beta,
                                  float *__restrict__ output,
                                  int row_size, const float eps)
{
    int id_row = blockIdx.x;
    int id_thread = threadIdx.x;
    int num_threads = blockDim.x;

    const float *row_input = input + id_row * row_size;
    float *row_output = output + id_row * row_size;

    float sum = 0.0f;
    float sq_sum = 0.0f;
    for (int i = id_thread; i < row_size; i += num_threads)
    {
        float val = row_input[i];
        sum += val;
        sq_sum += val * val;
    }

    extern __shared__ float shared[];
    shared[id_thread] = sum;
    shared[id_thread + num_threads] = sq_sum;
    __syncthreads();

    for (int stride = num_threads / 2; stride > 0; stride >>= 1)
    {
        if (id_thread < stride)
        {
            shared[id_thread] += shared[id_thread + stride];
            shared[id_thread + num_threads] += shared[id_thread + num_threads + stride];
        }
        __syncthreads();
    }

    float total_sum = shared[0];
    float total_sq_sum = shared[num_threads];

    __shared__ float mean;
    __shared__ float r_variance;
    if (id_thread == 0)
    {
        mean = total_sum / row_size;
        float variance = total_sq_sum / row_size - mean * mean;
        r_variance = rsqrtf(variance + eps);
    }
    __syncthreads();

    for (int i = id_thread; i < row_size; i += num_threads)
    {
        float normalized = (row_input[i] - mean) * r_variance;
        row_output[i] = normalized * gamma[i] + beta[i];
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

    d_input = cuda.mem_alloc(input_np.nbytes)
    d_gamma = cuda.mem_alloc(gamma_np.nbytes)
    d_beta = cuda.mem_alloc(beta_np.nbytes)
    d_output = cuda.mem_alloc(output.nbytes)

    cuda.memcpy_htod(d_input, input_np)
    cuda.memcpy_htod(d_gamma, gamma_np)
    cuda.memcpy_htod(d_beta, beta_np)

    mod = SourceModule(kernel_code)
    layer_norm_kernel = mod.get_function("layer_norm_kernel")

    threads_per_block = min(row_size, 256)
    shared_mem_size = threads_per_block * 8
    blocks_per_grid = input_np.size // row_size  
    layer_norm_kernel(d_input, d_gamma, d_beta, d_output, np.int32(row_size), np.float32(eps), block=(threads_per_block, 1, 1), grid = (blocks_per_grid, 1), shared=shared_mem_size)

    cuda.memcpy_dtoh(output, d_output)

    d_input.free()
    d_gamma.free()
    d_beta.free()
    d_output.free()

    return output
