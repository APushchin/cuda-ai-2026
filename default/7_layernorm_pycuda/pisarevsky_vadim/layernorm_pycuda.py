import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

BLOCK_SIZE = 256

layernorm_module = SourceModule(r'''
#define BLOCK_SIZE 256
#define WARP_SIZE 32
#define NUM_WARPS (BLOCK_SIZE / WARP_SIZE)

__device__ __forceinline__
void warp_reduce(float& s, float& sq) {
    #pragma unroll
    for (int k = WARP_SIZE/2; k > 0; k >>= 1) {
        s += __shfl_down_sync(0xffffffff, s, k);
        sq += __shfl_down_sync(0xffffffff, sq, k);
    }
}

__global__ void vecLayerNorm(const float* X, const float* gamma, const float* beta,
                             float* Y, int nrows, int ncols, float eps) {
    int row = blockIdx.x;
    if (row >= nrows) return;

    int tid = threadIdx.x;
    const float4* X4 = (const float4*)(X + row*ncols);
    const float4* G4 = (const float4*)gamma;
    const float4* B4 = (const float4*)beta;
    float4* Y4 = (float4*)(Y + row*ncols);
    int ncols_4 = ncols / 4;

    float t_sum = 0.f, t_sumsq = 0.f;
    for (int j = tid; j < ncols_4; j += BLOCK_SIZE) {
        float4 v = X4[j];
        t_sum += v.x + v.y + v.z + v.w;
        t_sumsq += v.x*v.x + v.y*v.y + v.z*v.z + v.w*v.w;
    }

    __shared__ float wsum[NUM_WARPS], wsumsq[NUM_WARPS];
    int lane = tid & (WARP_SIZE - 1), wid = tid / WARP_SIZE;

    warp_reduce(t_sum, t_sumsq);
    if (lane == 0) {
        wsum[wid] = t_sum;
        wsumsq[wid] = t_sumsq;
    }
    __syncthreads();

    if (wid == 0) {
        t_sum = lane < NUM_WARPS ? wsum[lane] : 0.f;
        t_sumsq = lane < NUM_WARPS ? wsumsq[lane] : 0.f;
        warp_reduce(t_sum, t_sumsq);
        if (lane == 0) { wsum[0] = t_sum; wsumsq[0] = t_sumsq; }
    }
    __syncthreads();
    float mean = wsum[0] / ncols;
    float var = fmaxf(wsumsq[0] / ncols - mean*mean, 0.f);
    float scale = rsqrtf(var + eps);

    for (int j = tid; j < ncols_4; j += BLOCK_SIZE) {
        float4 v = X4[j], g = G4[j], b = B4[j];
        float4 r = { g.x*(v.x - mean)*scale + b.x, g.y*(v.y - mean)*scale + b.y,
                     g.z*(v.z - mean)*scale + b.z, g.w*(v.w - mean)*scale + b.w };
        Y4[j] = r;
    }
}
''')
layernorm_kernel = layernorm_module.get_function('vecLayerNorm')


def layernorm_pycuda(input, gamma, beta, row_size, eps=1e-5):
    """
    Row-wise layer normalization: y = gamma * (x - mean) / sqrt(var + eps) + beta.
    """
    x = np.ascontiguousarray(input, dtype=np.float32)
    g = np.ascontiguousarray(gamma, dtype=np.float32)
    b = np.ascontiguousarray(beta, dtype=np.float32)
    y = np.empty_like(x)
    nrows = x.size // row_size

    assert (row_size % 4 == 0)

    d_x = cuda.mem_alloc(x.nbytes)
    d_g = cuda.mem_alloc(g.nbytes)
    d_b = cuda.mem_alloc(b.nbytes)
    d_y = cuda.mem_alloc(y.nbytes)
    cuda.memcpy_htod(d_x, x)
    cuda.memcpy_htod(d_g, g)
    cuda.memcpy_htod(d_b, b)

    layernorm_kernel(d_x, d_g, d_b, d_y,
                     np.int32(nrows), np.int32(row_size), np.float32(eps),
                     grid=(nrows, 1, 1), block=(BLOCK_SIZE, 1, 1))

    cuda.memcpy_dtoh(y, d_y)
    d_x.free()
    d_g.free()
    d_b.free()
    d_y.free()

    return y


def layernorm_numpy(input, gamma, beta, row_size, eps=1e-5):
    """Pure numpy reference with the same interface as layernorm_pycuda."""
    x = np.asarray(input, dtype=np.float32).reshape(-1, row_size)
    gamma = np.asarray(gamma, dtype=np.float32)
    beta = np.asarray(beta, dtype=np.float32)
    mean = x.mean(axis=1, keepdims=True)
    var = x.var(axis=1, keepdims=True)
    scale = 1./np.sqrt(var + np.float32(eps))
    y = gamma*((x - mean)*scale) + beta
    return y.reshape(-1)


if __name__ == "__main__":
    from time import perf_counter

    nrows, ncols = 8192, 16384
    rng = np.random.default_rng(0x1234)
    x = rng.uniform(-10., 10., nrows*ncols).astype(np.float32)
    gamma = rng.uniform(-10., 10., ncols).astype(np.float32)
    beta = rng.uniform(-10., 10., ncols).astype(np.float32)

    # Warming-up
    output = layernorm_pycuda(x, gamma, beta, ncols)

    t0 = perf_counter()
    outref = layernorm_numpy(x, gamma, beta, ncols)
    t_numpy = perf_counter() - t0

    nbad = int(np.count_nonzero(~np.isfinite(output)))
    for i in np.flatnonzero(~np.isfinite(output))[:100]:
        print(f"bad value {output[i]} at (row = {i // ncols}, col = {i % ncols})")
    err = float(np.abs(output - outref).max())
    print(f"max absolute error = {err:.5g}, {nbad} bad values")

    # Performance Measuring
    time_list = []
    for i in range(4):
        start = perf_counter()
        output = layernorm_pycuda(x, gamma, beta, ncols)
        time_list.append(perf_counter() - start)
    print(f"cuda time = {min(time_list):.4f} (including host<->device copying)")
    print(f"numpy time = {t_numpy:.4f}")
