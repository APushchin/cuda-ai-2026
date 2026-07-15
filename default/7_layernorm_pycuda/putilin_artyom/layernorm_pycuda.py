import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np
from pycuda.compiler import SourceModule

cLayerNormKernel = """

__global__ void calcMean(float* means, const float* input, int col_size, int row_size)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= col_size) return;

    float sum = 0.0f;

    for (int j = 0; j < row_size; ++j)
    {
        sum += input[i * row_size + j];
    }
    means[i] = sum / row_size;
}

__global__ void substractMean(float* in_out, int col_size, int row_size, float* mean)
{
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= col_size || j >= row_size) return;

    in_out[i * row_size + j] -= mean[i];
}

__global__ void calcVariance(float* mean, const float* input, int col_size, int row_size)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (j >= col_size) return;

    float varSum = 0.0f;
    float diff = 0.0f;

    for (int k = 0; k < row_size; ++k)
    {
        diff = input[j * row_size + k];
        varSum += diff * diff;
    }

    mean[j] = varSum / row_size;
}

__global__ void calcSqrt(float* input, int row_size, float eps)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= row_size) return;

    input[i] = 1.0f / sqrt(input[i] + eps);
}

__global__ void layerNormKernel(float* input, const float* sigma, const float* gamma, const float* beta,
                                            int col_size, int row_size, float eps)
{
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    const int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= col_size || j >= row_size) return;

    float x_upd = sigma[i] * gamma[j];
    input[i * row_size + j] = input[i * row_size + j] * x_upd + beta[j];
}
"""

module = SourceModule(cLayerNormKernel)
pyCalcMean = module.get_function("calcMean")
pySubstractMean = module.get_function("substractMean")
pyCalcVariance = module.get_function("calcVariance")
pyCalcSqrt = module.get_function("calcSqrt")
pyLayerNorm = module.get_function("layerNormKernel")

def layernorm_pycuda(input, gamma, beta, row_size, eps=1e-5):

    x = np.ascontiguousarray(input, dtype=np.float32)
    gamma_arr = np.ascontiguousarray(gamma, dtype=np.float32)
    beta_arr = np.ascontiguousarray(beta, dtype=np.float32)

    col_size = np.int32(x.size / row_size)
    row_size = np.int32(row_size)

    x_gpu = cuda.mem_alloc(x.nbytes)
    mean_gpu = cuda.mem_alloc(int(col_size * 4))
    gamma_gpu = cuda.mem_alloc(gamma.nbytes)
    beta_gpu = cuda.mem_alloc(beta.nbytes)

    # Copy from host 2 dev
    stream = cuda.Stream()

    cuda.memcpy_htod_async(x_gpu, x, stream)
    cuda.memcpy_htod_async(gamma_gpu, gamma_arr, stream)
    cuda.memcpy_htod_async(beta_gpu, beta_arr, stream)

    bs_vec = (256, 1, 1)
    nb_vec = (int((col_size + 255) // 256), 1)

    bs_mtrx = (16, 16, 1)
    nb_mtrx = (int((row_size + 15) // 16), int((col_size + 15) // 16),1)

    pyCalcMean(mean_gpu, x_gpu, col_size, row_size, block=bs_vec, grid=nb_vec, stream=stream)
    pySubstractMean(x_gpu, col_size, row_size, mean_gpu, block=bs_mtrx, grid=nb_mtrx, stream=stream)
    pyCalcVariance(mean_gpu, x_gpu, col_size, row_size, block=bs_vec, grid=nb_vec, stream=stream)
    pyCalcSqrt(mean_gpu, col_size, np.float32(eps), block=bs_vec, grid=nb_vec, stream=stream)
    pyLayerNorm(x_gpu, mean_gpu, gamma_gpu, beta_gpu, col_size, row_size, np.float32(eps),block=bs_mtrx, grid=nb_mtrx, stream=stream)

    y = np.empty_like(x)

    # Copy from dev 2 host
    cuda.memcpy_dtoh_async(y, x_gpu, stream)
    stream.synchronize()

    # Free memory
    x_gpu.free()
    mean_gpu.free()
    gamma_gpu.free()
    beta_gpu.free()

    return y
