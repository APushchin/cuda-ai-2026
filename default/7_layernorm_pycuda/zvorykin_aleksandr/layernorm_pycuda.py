import numpy as np

import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

cuda_cpp = """
__global__ void layernorm_fused(const float* input, float* output, const float* gamma, const float* beta, int m, int n, float eps) 
{
    int row = blockIdx.x;
    if (row >= m) return;
    int tId = threadIdx.x;
    int bSize = blockDim.x;
    __shared__ float sharedSum[256];
    __shared__ float sharedSqSum[256];
    float sum = 0.0f;
    for (int j = tId; j < n; j += bSize)
      sum += input[row * n + j];
    sharedSum[tId] = sum;
    __syncthreads();
    for (int s = bSize >> 1; s > 0; s >>= 1)
    {
        if (tId < s)
          sharedSum[tId] += sharedSum[tId + s];
        __syncthreads();
    }
    float mean = sharedSum[0] / n;
    float sqSum = 0.0f;
    for (int j = tId; j < n; j += bSize)
    {
        float x = input[row * n + j] - mean;
        sqSum += x * x;
    }
    sharedSqSum[tId] = sqSum;
    __syncthreads();
    for (int s = bSize >> 1; s > 0; s >>= 1)
    {
        if (tId < s) 
          sharedSqSum[tId] += sharedSqSum[tId + s];
        __syncthreads();
    }
    float var = sharedSqSum[0] / n;
    float inv_std = rsqrtf(var + eps);
    for (int j = tId; j < n; j += bSize)
    {
        float x = input[row * n + j];
        output[row * n + j] = gamma[j] * inv_std * (x - mean) + beta[j];
    }
}
"""

module = SourceModule(cuda_cpp)
kernel = module.get_function("layernorm_fused")

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
    # TODO: Implement using PyCUDA
    mat_cpu = np.asarray(input, dtype=np.float32)
    gamma_cpu = np.asarray(gamma, dtype=np.float32)
    beta_cpu = np.asarray(beta, dtype=np.float32)
  
    m = int(mat_cpu.size // row_size)
    n = row_size
  
    mat_gpu = cuda.mem_alloc(mat_cpu.nbytes)
    cuda.memcpy_htod(mat_gpu, mat_cpu)
  
    gamma_gpu = cuda.mem_alloc(gamma_cpu.nbytes)
    cuda.memcpy_htod(gamma_gpu, gamma_cpu)
  
    beta_gpu = cuda.mem_alloc(beta_cpu.nbytes)
    cuda.memcpy_htod(beta_gpu, beta_cpu)
  
    out_gpu = cuda.mem_alloc(mat_cpu.nbytes)
  
    block_dim = 256
    kernel(mat_gpu, out_gpu, gamma_gpu, beta_gpu, np.int32(m), np.int32(n), np.float32(eps), block=(block_dim, 1, 1), grid=(m, 1, 1), shared=2048)
    out = np.empty_like(mat_cpu)
    cuda.memcpy_dtoh(out, out_gpu)
    
    return out
