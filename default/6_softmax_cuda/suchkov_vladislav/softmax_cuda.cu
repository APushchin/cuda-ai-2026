#include "softmax_cuda.h"

#include <float.h>

#include <cuda/cmath>

#define BLOCK_SIZE 32

__global__ void SoftmaxCUDAKernel(const float* input, float* output, int num_cols) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    __shared__ float loc_maxs[BLOCK_SIZE];
    float loc_max = -__FLT_MAX__;
    for (int i = tid; i < num_cols; i += BLOCK_SIZE) {
        loc_max = fmaxf(loc_max, input[bid * num_cols + i]);
    }
    loc_maxs[tid] = loc_max;
    __syncthreads();

    __shared__ float row_max;
    if (tid == 0) {
        row_max = -__FLT_MAX__;
        for (int i = 0; i < BLOCK_SIZE; ++i) {
            row_max = fmax(loc_maxs[i], row_max);
        }
    }

    __shared__ float loc_sums[BLOCK_SIZE];
    float loc_sum = 0.f;
    for (int i = 0; i < num_cols; i += BLOCK_SIZE) {
        loc_sum += __expf(input[bid * num_cols + i] - row_max);
    }
    loc_sums[tid] = loc_sum;
    __syncthreads();

    __shared__ float row_sum;
    if (tid == 0) {
        row_sum = 0.f;
        for (int i = 0; i < BLOCK_SIZE; ++i) {
            row_sum += loc_sums[i];
        }
    }

    for (int i = 0; i < num_cols; i += BLOCK_SIZE) {
        int idx = bid + num_cols + i;
        output[idx] = __expf(input[idx] - row_max) / row_sum;
    }
}

std::vector<float> SoftmaxCUDA(const std::vector<float>& input, int row_count) {
    const int size = input.size() * sizeof(float);
    const int num_cols = input.size() / row_count;

    float* dev_ptr = nullptr;
    cudaMalloc(&dev_ptr, 2 * size);

    float* in_dev = dev_ptr;
    float* out_dev = dev_ptr + input.size();
    cudaMemcpy(in_dev, input.data(), size, cudaMemcpyHostToDevice);

    SoftmaxCUDAKernel<<<row_count, BLOCK_SIZE>>>(in_dev, out_dev, num_cols);

    std::vector<float> out(input.size());
    float* result = out.data();

    cudaDeviceSynchronize();
    cudaMemcpy(result, out_dev, size, cudaMemcpyDeviceToHost);
    cudaFree(dev_ptr);

    return out;
}
