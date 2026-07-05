#include "softmax_cuda.h"

#include <cuda_runtime.h>
#include <cfloat>
#include <vector>

constexpr int BLOCK_SIZE = 256;

__global__ void SoftmaxCUDA(const float* input,
                                float* output,
                                int rowSize) {
    const int row = blockIdx.x;
    const int tx = threadIdx.x;

    float rowMax = -FLT_MAX;
    for (int i = tx; i < rowSize; i += BLOCK_SIZE) {
        rowMax = fmaxf(rowMax, input[row * rowSize + i]);
    }

    __shared__ float buff[BLOCK_SIZE];
    buff[tx] = rowMax;

    __syncthreads();

    for (int idx = BLOCK_SIZE / 2; idx > 0; idx >>= 1) {
        if (tx < idx) {
            buff[tx] = fmaxf(buff[tx], buff[tx + idx]);
        }

        __syncthreads();
    }

    float rowSum = 0.f;
    for (int i = tx; i < rowSize; i += BLOCK_SIZE) {
        output[row * rowSize + i] = __expf(input[row * rowSize + i] - buff[0]);
        rowSum += output[row * rowSize + i];
    }

    buff[tx] = rowSum;

    __syncthreads();

    for (int idx = BLOCK_SIZE / 2; idx > 0; idx >>= 1) {
        if (tx < idx) {
            buff[tx] += buff[tx + idx];
        }
    
        __syncthreads();
    }

    for (int i = tx; i < rowSize; i += BLOCK_SIZE) {
        output[row * rowSize + i] *= (1.0f / buff[0]);
    }
}

std::vector<float> SoftmaxCUDA(const std::vector<float>& input,
                               int row_count) {
    const int size = input.size() * sizeof(float);
    const int rowSize = input.size() / row_count;

    float* gpu_in = nullptr;
    float* gpu_out = nullptr;

    cudaMalloc(&gpu_in, size);
    cudaMalloc(&gpu_out, size);

    cudaMemcpy(gpu_in, input.data(), size, cudaMemcpyHostToDevice);

    SoftmaxCUDA<<<row_count, BLOCK_SIZE>>>(gpu_in, gpu_out, rowSize);

    std::vector<float> output(input.size());

    cudaDeviceSynchronize();

    cudaMemcpy(output.data(), gpu_out, size, cudaMemcpyDeviceToHost);

    cudaFree(gpu_in);
    cudaFree(gpu_out);

    return output;
}
