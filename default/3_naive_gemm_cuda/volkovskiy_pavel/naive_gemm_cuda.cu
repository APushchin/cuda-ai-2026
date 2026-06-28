#include "naive_gemm_cuda.h"

#include <cuda/cmath>
#include <cuda_runtime.h>

__global__ void gemm_kernel(const float* in_a, float* in_b, float* out, size_t n)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < n && j < n) {
        float sum = 0.0f;
        for (size_t k = 0; k < n; ++k)
            sum += in_a[i * n + k] * in_b[k * n + j];
        out[i * n  + j] = sum;
    }
}

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n)
{
    const std::size_t memsize = a.size() * sizeof(float);

    static int minGridSize = 0;
    static int maxBlockSize = 0;
    static bool isBlockSizeComputed = false;
    if (!isBlockSizeComputed) {
        isBlockSizeComputed = true;
        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &maxBlockSize, gemm_kernel);
    }

    dim3 threadsPerBlock(32, maxBlockSize / 32);

    dim3 numBlocks(cuda::ceil_div((unsigned)n, threadsPerBlock.x),
                   cuda::ceil_div((unsigned)n, threadsPerBlock.y));

    float *in_a = nullptr;
    float *in_b = nullptr;
    float *out = nullptr;

    cudaMalloc((void**)&in_a, memsize);
    cudaMalloc((void**)&in_b, memsize);
    cudaMalloc((void**)&out, memsize);

    cudaMemcpy(in_a, a.data(), memsize, cudaMemcpyHostToDevice);
    cudaMemcpy(in_b, b.data(), memsize, cudaMemcpyHostToDevice);

    gemm_kernel<<<numBlocks, threadsPerBlock>>>(in_a, in_b, out, n);

    std::vector<float> result(a.size());

    cudaMemcpy(result.data(), out, memsize, cudaMemcpyDeviceToHost);

    cudaFree(in_a);
    cudaFree(in_b);
    cudaFree(out);

    return result;
}