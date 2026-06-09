#include "naive_gemm_cuda.h"

#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <cuda/cmath>

__global__ void naiveGemmKernel(const float *a, const float *b, float *c, int n)
{

    int icol = blockIdx.x * blockDim.x + threadIdx.x;
    int irow = blockIdx.y * blockDim.y + threadIdx.y;

    if (irow < n && icol < n)
    {
        float sum = 0.0f;
        for (int k = 0; k < n; ++k)
        {
            sum += a[irow * n + k] * b[k * n + icol];
        }
        c[irow * n + icol] = sum;
    }
}

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {

    size_t N = n *n;
    size_t size = N  * sizeof(float);
    std::vector<float> c(N);

    float *dev_a = nullptr;
    float *dev_b = nullptr;
    float *dev_c = nullptr;

    float *host_a = const_cast<float *>(a.data());
    float *host_b = const_cast<float *>(b.data());
    float *host_c = const_cast<float *>(c.data());

    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);

    cudaMalloc(&dev_a, size);
    cudaMalloc(&dev_b, size);
    cudaMalloc(&dev_c, size);

    cudaHostRegister(const_cast<void *>(static_cast<const void *>(host_a)), size, cudaHostRegisterDefault);
    cudaHostRegister(const_cast<void *>(static_cast<const void *>(host_b)), size, cudaHostRegisterDefault);
    cudaHostRegister(const_cast<void *>(static_cast<const void *>(host_c)), size, cudaHostRegisterDefault);

    cudaMemcpyAsync(dev_a, host_a, size, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(dev_b, host_b, size, cudaMemcpyHostToDevice, stream);

    //const int blockSize = 256;

    int blockSize, minGridSize;
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, (void *)naiveGemmKernel, 0, N);

    dim3 threads(blockSize, blockSize);
    int blocksNum = cuda::ceil_div(n,blockSize);
    dim3 blocks(blocksNum, blocksNum);

    naiveGemmKernel<<<blocks, threads, 0, stream>>>(dev_a, dev_b, dev_c, n);

    cudaDeviceSynchronize();

    cudaMemcpyAsync(host_c, dev_c, size, cudaMemcpyDeviceToHost, stream);

    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    return c;
    
}