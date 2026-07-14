import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np

kernel_code = """
__global__ void layerNormFunc(float *output, float *input, float *gamma, float *beta, int row_size, int col_size, int num_mtx_el, float eps)
{
    extern __shared__ float sharedMemory[];

    float* sharedMean = sharedMemory;
    float* sharedInvVar  = sharedMemory[col_size];
    float* sharedOther  = sharedMemory[2*col_size];

    int index = threadIdx.x + blockIdx.x * blockDim.x;

    if (index < col_size)
    {
        const int elStart = index * row_size;
        sharedMean[index] = 0;
        for (int iRowEl = 0; iRowEl < row_size; ++iRowEl)
            sharedMean[index] += input[elStart + iRowEl];
        sharedMean[index] /= row_size;
    }

    __syncthreads();

    if (index < num_mtx_el)
    {
        int meanIndex = index/row_size;
        sharedOther[index] = (input[index] - sharedMean[meanIndex]) * (input[index] - sharedMean[meanIndex]);
    }

    __syncthreads();

    if (index < col_size)
    {
        const int elStart = index * row_size;
        sharedInvVar[index] = 0;
        for (int iRowEl = 0; iRowEl < row_size; ++iRowEl)
            sharedInvVar[index] += sharedOther[elStart + iRowEl];
        sharedInvVar[index] /= row_size;
        sharedInvVar[index] = 1 / rsqrtf(sharedInvVar[index] + eps);
    }

    __syncthreads();

    if (index < num_mtx_el)
    {
        int rowIndex  = index/row_size;
        output[index] = gamma[index] * (input[index] - sharedMean[rowIndex]) * sharedInvVar[rowIndex] + beta[index];
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

    threads_per_block = 256
    blocks_per_grid = (input.size + threads_per_block - 1) // threads_per_block
    shared_mem_size = input.size + 2*col_size

    col_size = input.size / row_size

    layernorm_kernel(device_output, device_input, device_gamma, device_beta, np.int32(row_size), np.int32(col_size), np.int32(input.size), np.float32(eps),
    block=(threads_per_block, 1, 1),
    grid=(blocks_per_grid, 1),
    shared=shared_mem_size)

    output = np.empty_like(input)
    cuda.memcpy_dtoh(output, device_output)

    device_input.free()
    device_gamma.free()
    device_beta.free()
    device_output.free()

    return output