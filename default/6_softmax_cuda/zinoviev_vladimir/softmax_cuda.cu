#include <algorithm>
#include <chrono>
#include <vector>
#include <iostream>
#include <random>
#include <cuda_runtime.h>
#include "softmax_cuda.h"

#define BLOCK_SIZE 256

__device__ inline void merge(float max_a, float sum_a,
                             float max_b, float sum_b,
                             volatile float& out_max, volatile float& out_sum) {
    out_max = fmaxf(max_a, max_b);
    out_sum = sum_a * expf(max_a - out_max) 
            + sum_b * expf(max_b - out_max);
}

__global__ void SoftmaxCUDAKernel(const float* input, float* output, int col_size) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    float local_max = -__FLT_MAX__;
    float local_sum = 0.f;
    const float4* input4 = reinterpret_cast<const float4*>(input + bid * col_size);
    float4* output4 = reinterpret_cast<float4*>(output + bid * col_size);

    for (int i = tid; i < col_size / 4; i += BLOCK_SIZE) {
        float4 x = input4[i];
        float new_max = fmaxf(local_max, x.x);
        new_max = fmaxf(new_max, x.y);
        new_max = fmaxf(new_max, x.z);
        new_max = fmaxf(new_max, x.w);
        local_sum = local_sum * expf(local_max - new_max) +
            expf(x.x - new_max) +
            expf(x.y - new_max) +
            expf(x.z - new_max) +
            expf(x.w - new_max);
        local_max = new_max;
    }

    __shared__ float smax[BLOCK_SIZE];
    __shared__ float ssum[BLOCK_SIZE];

    smax[tid] = local_max;
    ssum[tid] = local_sum;
    __syncthreads();

    for (int s = BLOCK_SIZE / 2; s > 32; s >>= 1) {
        if (tid < s)
            merge(smax[tid], ssum[tid],
                  smax[tid + s], ssum[tid + s],
                  smax[tid], ssum[tid]);
        __syncthreads();
    }

    if (tid < 32) {
        volatile float* vmax = smax;
        volatile float* vsum = ssum;
        merge(vmax[tid], vsum[tid], vmax[tid+32], vsum[tid+32], vmax[tid], vsum[tid]); __syncwarp();
        merge(vmax[tid], vsum[tid], vmax[tid+16], vsum[tid+16], vmax[tid], vsum[tid]); __syncwarp();
        merge(vmax[tid], vsum[tid], vmax[tid+ 8], vsum[tid+ 8], vmax[tid], vsum[tid]); __syncwarp();
        merge(vmax[tid], vsum[tid], vmax[tid+ 4], vsum[tid+ 4], vmax[tid], vsum[tid]); __syncwarp();
        merge(vmax[tid], vsum[tid], vmax[tid+ 2], vsum[tid+ 2], vmax[tid], vsum[tid]); __syncwarp();
        merge(vmax[tid], vsum[tid], vmax[tid+ 1], vsum[tid+ 1], vmax[tid], vsum[tid]); __syncwarp();
    }
    __syncthreads();

    float row_max = smax[0];
    float row_sum = ssum[0];
    for (int i = tid; i < col_size / 4; i += BLOCK_SIZE) {
        output4[i] = make_float4(
            __expf(input4[i].x - row_max) / row_sum,
            __expf(input4[i].y - row_max) / row_sum,
            __expf(input4[i].z - row_max) / row_sum,
            __expf(input4[i].w - row_max) / row_sum
        );
    }
}

class SoftmaxCUDAHandler {
public:
    SoftmaxCUDAHandler() : d_in(nullptr), d_out(nullptr), memSizeLast(0) {
        cudaStreamCreate(&stream);
    }

    std::vector<float>& execute(const std::vector<float>& input, int row_size) {
        const int col_size = input.size() / row_size;
        const size_t memSize = input.size() * sizeof(float);
        if (memSize > memSizeLast) {
            if (d_in) {
                cudaFree(d_in);
                cudaFree(d_out);
            }
            cudaMalloc(&d_in, memSize);
            cudaMalloc(&d_out, memSize);
            output = std::vector<float>(input.size());
            memSizeLast = memSize;
        }

        cudaMemcpyAsync(this->d_in, input.data(), memSize, cudaMemcpyHostToDevice, stream);
        SoftmaxCUDAKernel<<<row_size, BLOCK_SIZE>>>(d_in, d_out, col_size);
        cudaMemcpyAsync(output.data(), d_out, memSize, cudaMemcpyDeviceToHost, stream);

        cudaStreamSynchronize(stream);
        return output;
    }

    ~SoftmaxCUDAHandler() {
        cudaFree(d_in);
        cudaFree(d_out);

        cudaStreamDestroy(stream);
    }
private:
    cudaStream_t stream;
    std::vector<float> output;
    float *d_in, *d_out;
    size_t memSizeLast;
};

std::vector<float> SoftmaxCUDA(const std::vector<float>& input, int row_size) {
    static SoftmaxCUDAHandler handler;
    return handler.execute(input, row_size);
}
