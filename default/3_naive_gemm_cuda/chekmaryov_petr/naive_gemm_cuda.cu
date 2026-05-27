#include "naive_gemm_cuda.h"
#include <assert.h>
#include <stdio.h>

// CUDA-ядро для параллельного сложения элементов
__global__ void gemmKernel(float* a_ptr, float* b_ptr, float* res_ptr, int n)
{
    int i = threadIdx.y + blockIdx.y * blockDim.y;
    int j = threadIdx.x + blockIdx.x * blockDim.x;
    if(i < n && j < n)
    {
        float sum = 0;
        for(int k = 0; k < n; k++)
            sum += a_ptr[i * n + k] * b_ptr[k * n + j];
        res_ptr[i * n + j] = sum;
    }
}


std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n)
{
    assert(a.size() == n * n);
    assert(b.size() == n * n);
    const size_t size = n * n * sizeof(float);
    std::vector<float> res(n*n,0);

    float* a_ptr;
    float* b_ptr;
    float* res_ptr;
    cudaMalloc(&a_ptr, size);
    cudaMalloc(&b_ptr, size);
    cudaMalloc(&res_ptr, size);

    cudaMemcpy(a_ptr, a.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(b_ptr, b.data(), size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 blockCount((n+16-1) / 16, (n+16-1) / 16);
    gemmKernel<<<blockCount, threadsPerBlock>>>(a_ptr, b_ptr, res_ptr, n);
    cudaMemcpy(res.data(), res_ptr, size, cudaMemcpyDeviceToHost);
    cudaFree(a_ptr);
    cudaFree(b_ptr);
    cudaFree(res_ptr);
    
    return res;
}