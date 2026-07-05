#include "block_gemm_cuda.h"

#include <cuda/cmath>

#define BLOCK_X 16
#define BLOCK_SIZE BLOCK_X * BLOCK_X

__global__ void BlockGemmCUDAKernel(const float* a, const float* b, float* c, int n) {
    __shared__ float block_A[BLOCK_SIZE];
    __shared__ float block_B[BLOCK_SIZE];

    int x = blockIdx.x * BLOCK_X + threadIdx.x;
    int y = blockIdx.y * BLOCK_X + threadIdx.y;
    int shared_idx = threadIdx.y * BLOCK_X + threadIdx.x;

    float acc = 0.f;
    for (int blk = 0; blk < gridDim.x; ++blk) {
        block_A[shared_idx] = a[threadIdx.x + y * n + blk * BLOCK_X];
        block_B[shared_idx] = b[(threadIdx.y + blk * BLOCK_X) * n + x];
        __syncthreads();

        for (int k = 0; k < BLOCK_X; ++k) {
            acc += block_A[threadIdx.y * BLOCK_X + k] * block_B[k * BLOCK_X + threadIdx.x];
        }
        __syncthreads();
    }
    c[x + y * n] = acc;
}

std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    const int size = a.size() * sizeof(float);
    
    float* dev_ptr = nullptr;
    cudaMalloc(&dev_ptr, 3 * size);

    float* a_dev = dev_ptr;
    float* b_dev = dev_ptr + a.size();
    float* c_dev = dev_ptr + 2 * a.size();

    cudaMemcpy(a_dev, a.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(b_dev, b.data(), size, cudaMemcpyHostToDevice);
    cudaMemset(c_dev, 0, size);

    int num_blocks = n / BLOCK_X;
    BlockGemmCUDAKernel<<<{num_blocks, num_blocks}, {BLOCK_X, BLOCK_X}>>>(a_dev, b_dev, c_dev, n);

    std::vector<float> c(a.size());
    float* res = c.data();
    
    cudaDeviceSynchronize();

    cudaMemcpy(res, c_dev, size, cudaMemcpyDeviceToHost);
    cudaFree(dev_ptr);

    return c;
}
