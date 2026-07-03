import numpy as np
import math
from numba import cuda, float32

max_threads_per_block = 256
warp_size = 32
threads_per_block_x = warp_size

@cuda.jit(cache=True)
def norm_kernel(output, input, nrow, ncol, gamma, beta, eps):

    x, y = cuda.grid(2)
    
    warp_tid = cuda.threadIdx.x
    block_y = cuda.threadIdx.y
    stride = cuda.blockDim.x

    shared_sum = cuda.shared.array(threads_per_block_x, dtype=float32)

    one_over_n = 1.0 / ncol;

    if y < nrow:
        row_offset = y * ncol
        
        # sum each row
        sum = 0.0
        i = warp_tid;
        while i < ncol:
            sum += input[row_offset + i]
            i += stride
        
        shared_sum[cuda.blockDim.x  * cuda.threadIdx.y  + cuda.threadIdx.x] = sum;

        cuda.syncthreads();

        sum = 0.0;
        i = 0;
        while i < stride:
            sum += shared_sum[cuda.blockDim.x  * cuda.threadIdx.y  + i];
            i += 1
        
        u = sum / ncol

        #compute out = (x_i - u) and sum2 = (x_i -u)^2

        sum2 = 0.0
        i = warp_tid;
        while i < ncol:
            val = input[row_offset + i] - u
            sum2 += val * val
            output[row_offset + i] = val
            i += stride
        
        shared_sum[cuda.blockDim.x  * cuda.threadIdx.y  + cuda.threadIdx.x] = sum2;
        cuda.syncthreads();

        sum2 = 0.0;
        i = 0;
        while i < stride:
            sum2 += shared_sum[cuda.blockDim.x  * cuda.threadIdx.y  + i];
            i += 1

        sigma2 = sum2 * one_over_n

        one_over_sqrt_simga_eps = 1.0 / math.sqrt(sigma2 + eps)

        i = warp_tid;
        while i < ncol:
            output[row_offset + i] = output[row_offset + i] * gamma[i] * one_over_sqrt_simga_eps + beta[i]
            i += stride

    return;


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

    ncol = row_size
    nrow = input.size / ncol

    threads_per_block = (warp_size, math.ceil(max_threads_per_block / warp_size))

    blocks_per_grid = (1, math.ceil(nrow / threads_per_block[1]))

    device_input = cuda.to_device(input)
    device_gamma = cuda.to_device(gamma)
    device_beta = cuda.to_device(beta)
    device_output = cuda.device_array_like(input)

    norm_kernel[blocks_per_grid, threads_per_block](device_output, device_input, nrow, ncol, device_gamma, device_beta, eps)
    result = device_output.copy_to_host()

    return result.reshape(input.shape)