import cupy as cp
import numpy as np


layernorm_kernel = cp.RawKernel(r'''
__device__ __forceinline__ float warpReduceSum(float val) {
    for (int mask = 16; mask > 0; mask /= 2) {
        val += __shfl_down_sync(0xffffffff, val, mask);
    }
    return val;
}

__device__ __forceinline__ float blockReduceSum(float val) {
    __shared__ float shared_val[32]; 
    
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;

    val = warpReduceSum(val);

    if (lane == 0) shared_val[wid] = val;
    __syncthreads();

    int num_warps = (blockDim.x + 31) >> 5;
    float final_sum = (lane < num_warps) ? shared_val[lane] : 0.0f;

    if (wid == 0) final_sum = warpReduceSum(final_sum);
    __syncthreads();

    return final_sum;
}

extern "C" __global__
void layernorm(const float* x, const float* gamma, const float* beta, 
               float* y, int row_size, float eps) {
    __shared__ float s_mean;
    __shared__ float s_inv_std;

    int row_start = blockIdx.x * row_size;

    float local_sum = 0.0f;
    float local_sq_sum = 0.0f;

    const float4* x4 = (const float4*)(x + row_start);

    for (int j = threadIdx.x; j < row_size / 4; j += blockDim.x) {
        float4 v4 = x4[j];
        local_sum += v4.x + v4.y + v4.z + v4.w;
        local_sq_sum += v4.x * v4.x + v4.y * v4.y + v4.z * v4.z + v4.w * v4.w;
    }

    float total_sum = blockReduceSum(local_sum);
    float total_sq_sum = blockReduceSum(local_sq_sum);

    if (threadIdx.x == 0) {
        float mean = total_sum / row_size;
        float var = (total_sq_sum / row_size) - (mean * mean);
        s_mean = mean;
        s_inv_std = rsqrtf(max(0.0f, var) + eps);
    }
    __syncthreads();

    float mean = s_mean;
    float inv_std = s_inv_std;

    float4* y4 = (float4*)(y + row_start);
    const float4* gamma4 = (const float4*)gamma;
    const float4* beta4 = (const float4*)beta;

    for (int j = threadIdx.x; j < row_size / 4; j += blockDim.x) {
        float4 v4 = x4[j];
        float4 g4 = gamma4[j];
        float4 b4 = beta4[j];
        
        float4 r4;
        r4.x = g4.x * ((v4.x - mean) * inv_std) + b4.x;
        r4.y = g4.y * ((v4.y - mean) * inv_std) + b4.y;
        r4.z = g4.z * ((v4.z - mean) * inv_std) + b4.z;
        r4.w = g4.w * ((v4.w - mean) * inv_std) + b4.w;

        y4[j] = r4;
    }
}
''', 'layernorm')


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
    x = cp.asarray(input)
    g = cp.asarray(gamma)
    b = cp.asarray(beta)
    y = cp.empty_like(x)

    threads = 1024
    blocks = input.size // row_size
    layernorm_kernel((blocks,), (threads,), (x, g, b, y, row_size, eps))

    return cp.asnumpy(y)