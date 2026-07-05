#include "block_gemm_cuda.h"

#include <cuda/cmath>
#include <cuda_runtime.h>

static constexpr size_t BS = 16;

__global__ void gemm_kernel(const float* in_a, float* in_b, float* out, size_t n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int j = threadIdx.x;
    int i = threadIdx.y;
    __shared__ float a[BS * BS];
    __shared__ float b[BS * BS];
    if (idx < n && idy < n) {
        float sum = 0.0f;
        for (int t = 0; t < gridDim.x ; ++t) {
            a[BS * i + j] = in_a[idy * n + (BS * t + j)];
            b[BS * i + j] = in_b[(BS * t + i) * n + idx];
            __syncthreads();

            for (size_t l = 0; l < BS && (t * BS + l) < n; ++l)
                sum += a[i * BS + l] * b[l * BS + j];
            __syncthreads();
        }
        out[idy * n  + idx] = sum;
    }
}

std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n)
{
    const std::size_t memsize = a.size() * sizeof(float);

    dim3 threadsPerBlock(BS, BS);

    dim3 numBlocks((n + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (n + threadsPerBlock.y - 1) / threadsPerBlock.y);


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