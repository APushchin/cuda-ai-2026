import pycuda.autoinit
import pycuda.driver as drv
import numpy as np
from pycuda.compiler import SourceModule

cLayerNormKernel = """

extern "C" __global__ void layerNormKernel(const float* x, float* y, const float* gamma, const float* beta,
                                            int row_size, float eps)
{
    int row = blockIdx.x;
    int tid = threadIdx.x;

    int row_offset = row * row_size;

    float sum = 0.0f;
    for (int i = tid; i < row_size; i += blockDim.x) {
        sum += x[row_offset + i];
    }

    __shared__ float s_mean;
    if (tid == 0) s_mean = 0.0f;
    __syncthreads();
    atomicAdd(&s_mean, sum);
    __syncthreads();

    float mean = s_mean / row_size;

    float var_sum = 0.0f;
    for (int i = tid; i < row_size; i += blockDim.x) {
        float diff = x[row_offset + i] - mean;
        var_sum += diff * diff;
    }

    __shared__ float s_var;
    if (tid == 0) s_var = 0.0f;
    __syncthreads();
    atomicAdd(&s_var, var_sum);
    __syncthreads();

    float variance = s_var / row_size;
    float rsqrt_var = rsqrtf(variance + eps);

    for (int i = tid; i < row_size; i += blockDim.x) {
        int idx = row_offset + i;
        float normalized = (x[idx] - mean) * rsqrt_var;
        y[idx] = normalized * gamma[i] + beta[i];
    }
}
"""

module = SourceModule(cLayerNormKernel)
pyLayerNorm = module.get_function("layerNormKernel")

def layernorm_pycuda(x, gamma, beta, row_size, eps=1e-5):

    x = np.asarray(x, dtype=np.float32)
    gamma = np.asarray(gamma, dtype=np.float32)
    beta = np.asarray(beta, dtype=np.float32)

    batch_size, row = x.shape

    x_gpu = drv.mem_alloc(x.nbytes)
    y_gpu = drv.mem_alloc(x.nbytes)
    gamma_gpu = drv.mem_alloc(gamma.nbytes)
    beta_gpu = drv.mem_alloc(beta.nbytes)

    drv.memcpy_htod(x_gpu, x)
    drv.memcpy_htod(gamma_gpu, gamma)
    drv.memcpy_htod(beta_gpu, beta)

    block_size = min(row, 512)
    grid_size = batch_size

    pyLayerNorm(x_gpu, y_gpu, gamma_gpu, beta_gpu, np.int32(row), np.float32(eps),
        block=(block_size, 1, 1), grid=(grid_size, 1))

    y = np.empty_like(x)
    drv.memcpy_dtoh(y, y_gpu)

    return y

# --- Test ---
#batch_size = 8
#row_size = 8

#x_test = np.full((batch_size, row_size),300)
#x_test = [[i + j * row_size for i in range(row_size)] for j in range(row_size)]
#gamma_test = np.ones(row_size, dtype=np.float32)
#beta_test = np.zeros(row_size, dtype=np.float32)

#y_out = layernorm_pycuda(x_test, gamma_test, beta_test, row_size)

#print(y_out)
#print(y_out)
#print(y_out)
