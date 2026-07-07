#include "softmax_cuda.h"
#include <cuda_runtime.h>

#include <cmath>
#include <iostream>
#include <thread>
#include <cfloat>

constexpr int WARP_SIZE = 32;

__device__ float warpReduceMax(float val)
{
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
    {
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

__device__ float warpReduceSum(float val)
{
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
    {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

__global__ void softMaxExecute(const float *__restrict__ input, float *__restrict__ output, 
                                int row_count, int row_size)
{
    __shared__ float shared[WARP_SIZE];
    
    int row_id = blockIdx.x;
    int thread_id = threadIdx.x;
    int threads_per_block = blockDim.x;
    int row_start = row_id * row_size;

    int warp_id = thread_id / WARP_SIZE;
    int lane_id = thread_id % WARP_SIZE;
    int num_warps = (threads_per_block + WARP_SIZE - 1) / WARP_SIZE;

    float local_max = -FLT_MAX;
    for (int i = thread_id; i < row_size; i += threads_per_block)
    {
        local_max = fmaxf(local_max, input[row_start + i]);
    }
    local_max = warpReduceMax(local_max);

    if (lane_id == 0)
    {
        shared[warp_id] = local_max;
    }
    __syncthreads();

    if (warp_id == 0)
    {
        float max = (lane_id < num_warps) ? shared[lane_id] : -FLT_MAX;
        max = warpReduceMax(max);
        if (lane_id == 0)
        {
            shared[0] = max;
        }
    }
    __syncthreads();

    float row_max = shared[0];

    float local_sum = 0.0f;
    float exp[WARP_SIZE];
    int exp_count = 0;
    for (int i = thread_id; i < row_size; i += threads_per_block)
    {
        float val = __expf(input[row_start + i] - row_max);
        exp[exp_count++] = val;
        local_sum += val;
    }
    local_sum = warpReduceSum(local_sum);

    if (lane_id == 0)
    {
        shared[warp_id] = local_sum;
    }
    __syncthreads();

    if (warp_id == 0)
    {
        float cross_sum = (lane_id < num_warps) ? shared[lane_id] : 0.0f;
        cross_sum = warpReduceSum(cross_sum);
        if (lane_id == 0)
        {
            shared[0] = cross_sum;
        }
    }
    __syncthreads();

    float row_sum = shared[0];

    float inv_row_sum = 1.0f / row_sum;
    exp_count = 0;
    for (int col = thread_id; col < row_size; col += threads_per_block)
    {
        output[row_start + col] = exp[exp_count++] * inv_row_sum;
    }
}

std::vector<float> SoftmaxCUDA(const std::vector<float> &input, int row_count)
{
    const size_t size = input.size();

    std::vector<float> output;
    std::thread t([&]()
                  { output.resize(size); });

    const size_t bytes = size * sizeof(float);
    const int row_size = size / row_count;

    float *inputBuffer = nullptr;
    float *outputBuffer = nullptr;

    cudaMalloc(&inputBuffer, bytes);
    cudaMalloc(&outputBuffer, bytes);

    cudaMemcpy(inputBuffer, input.data(), bytes, cudaMemcpyHostToDevice);

    const int blockSize = 256;
    int numBlocks = row_count;
    softMaxExecute<<<numBlocks, blockSize>>>(inputBuffer, outputBuffer, row_count, row_size);
    cudaDeviceSynchronize();

    t.join();
    cudaMemcpy(output.data(), outputBuffer, bytes, cudaMemcpyDeviceToHost);

    cudaFree(inputBuffer);
    cudaFree(outputBuffer);

    return output;
}