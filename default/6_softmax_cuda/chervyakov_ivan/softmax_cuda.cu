#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <cuda/cmath>
#include <math_constants.h>
#include <float.h>
#include <thread>

#include "softmax_cuda.h"

constexpr int BLOCK_SIZE_SM_K = 256;

__device__ float reduce_max(float value, float *buffer)
{
    buffer[threadIdx.x] = value;
    __syncthreads();

    for (unsigned s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (threadIdx.x < s)
        {
            float left = buffer[threadIdx.x];
            float right = buffer[threadIdx.x + s];
            buffer[threadIdx.x] = (left < right) ? right : left;
        }
        __syncthreads();
    }

    return buffer[0];
}

__device__ float reduce_sum(float value, float *buffer)
{
    buffer[threadIdx.x] = value;
    __syncthreads();

    for (unsigned s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (threadIdx.x < s)
        {
            buffer[threadIdx.x] += buffer[threadIdx.x + s];
        }
        __syncthreads();
    }

    return buffer[0];
}

__global__ void SoftmaxKernel(const float *input, float *output, int row_size, int col_size)
{
    __shared__ float buffer[BLOCK_SIZE_SM_K];

    // Step 1: Find local maxima
    float local_max = -INFINITY;

    for (int i = threadIdx.x; i < row_size; i += blockDim.x)
    {
        float value = input[blockIdx.x * row_size + i];
        if (value > local_max)
            local_max = value;
    }

    float max_val = reduce_max(local_max, buffer);

    __syncthreads();

    // Step 2: E^(x - max)
    float local_sum = 0.0f;

    for (int i = threadIdx.x; i < row_size; i += blockDim.x)
    {
        float value = input[blockIdx.x * row_size + i];
        local_sum += expf(value - max_val);
    }

    float sum_val = reduce_sum(local_sum, buffer);

    __syncthreads();

    // Step 3: E^x / SUM
    for (int i = threadIdx.x; i < row_size; i += blockDim.x)
    {
        float value = input[blockIdx.x * row_size + i];
        output[blockIdx.x * row_size + i] = expf(value - max_val) / sum_val;
    }
}

std::vector<float> SoftmaxCUDA(const std::vector<float> &input, int row_count)
{
    const int data_size = input.size();
    std::vector<float> output;
    std::thread t([&output, data_size](){ output.resize(data_size); });

    const int row_size = data_size / row_count;
    size_t mem_size = data_size * sizeof(float);

    const float *input_data = input.data();

    float *devInput = nullptr;
    float *devOutput = nullptr;
    cudaMalloc(&devInput, mem_size);
    cudaMalloc(&devOutput, mem_size);

    cudaMemcpy(devInput, input_data, mem_size, cudaMemcpyHostToDevice);

    SoftmaxKernel<<<row_count, BLOCK_SIZE_SM_K>>>(devInput, devOutput, row_count, row_size);

    t.join();

    cudaMemcpy(output.data(), devOutput, mem_size, cudaMemcpyDeviceToHost);

    cudaFree(devInput);
    cudaFree(devOutput);

    return output;
}