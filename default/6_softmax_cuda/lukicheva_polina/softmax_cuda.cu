#include "sofrowCount_cuda.h"

#include <cuda/cmath>
#include <chrono>
#include <vector>
#include <iostream>
#include <algorithm>
#include <float.h>
#include <thread>

#define WARP_SIZE 32
#define BLOCK_SIZE 256

__global__ void SofrowCountCUDAKernel(float* input, int rowCount, int rowSize) {
    int rawIdx = blockIdx.x;
    if (rawIdx >= rowCount) {
        return;
    }

    int tIdx = threadIdx.x;
    int warp_id = tIdx / WARP_SIZE;
    int lane_id = tIdx % WARP_SIZE;
    
    float* row_input_data = input + rawIdx * rowSize;

    float t_max = -FLT_MAX;
    for (int col = tIdx; col < rowSize; col += blockDim.x) {
        t_max = fmaxf(t_max, row_input_data[col]);
    }

    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        t_max = fmaxf(t_max, __shfl_down_sync(0xFFFFFFFF, t_max, offset));
    }

    __shared__ float shared_mem[WARP_SIZE]; 
    if (lane_id == 0) {
        shared_mem[warp_id] = t_max;
    }
    __syncthreads();

    float global_max = (tIdx < (blockDim.x / WARP_SIZE)) ? shared_mem[lane_id] : -INFINITY;
    if (warp_id == 0) {
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            global_max = fmaxf(global_max, __shfl_down_sync(0xFFFFFFFF, global_max, offset));
        }
        if (tIdx == 0) {
            shared_mem[0] = global_max;
        }
    }
    __syncthreads();
    global_max = shared_mem[0];

    float t_sum = 0.0f;
    for (int col = tIdx; col < rowSize; col += blockDim.x) {
        float exp_val = expf(row_input_data[col] - global_max);
        row_input_data[col] = exp_val;
        t_sum += exp_val;
    }

    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        t_sum += __shfl_down_sync(0xFFFFFFFF, t_sum, offset);
    }

    if (lane_id == 0) {
        shared_mem[warp_id] = t_sum;
    }
    __syncthreads();

    float global_sum = (tIdx < (blockDim.x / WARP_SIZE)) ? shared_mem[lane_id] : 0.0f;
    if (warp_id == 0) {
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            global_sum += __shfl_down_sync(0xFFFFFFFF, global_sum, offset);
        }
        if (tIdx == 0) {
            shared_mem[0] = global_sum;
        }
    }
    __syncthreads();
    
    float k = 1.0f / shared_mem[0];
    for (int col = tIdx; col < rowSize; col += blockDim.x) {
        row_input_data[col] *= k;
    }
}

std::vector<float> SoftmaxCUDA(const std::vector<float>& input, int rowCount) {
    const int data_size = input.size();
    std::vector<float> output;
    std::thread t([&output, data_size](){
        output.resize(data_size);
    });
    const int rowSize = data_size / rowCount;
    const float* input_data = input.data();

    float* devInput = nullptr;
    cudaMalloc(&devInput, data_size * sizeof(float));

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaMemcpyAsync(devInput, input_data, data_size * sizeof(float), cudaMemcpyHostToDevice, stream);

    SofrowCountCUDAKernel<<<rowCount, BLOCK_SIZE, 0, stream>>>(devInput, rowCount, rowSize);

    t.join();
    cudaMemcpyAsync(output.data(), devInput, data_size * sizeof(float), cudaMemcpyDeviceToHost, stream);

    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);

    cudaFree(devInput);
    return output;
}

#if 0
std::vector<float> SoftmaxRef(const std::vector<float>& input, int row_size) {
    size_t col_size = input.size() / row_size;
    std::vector<float> output(row_size * col_size);

    const float* inptr = input.data();
    float* outptr = output.data();

    for (size_t i = 0; i < row_size; i++) {
        const float* row_in = inptr + i * col_size;
        float* row_out = outptr + i * col_size;

        float max = std::numeric_limits<float>::lowest();
        for (size_t j = 0; j < col_size; j++) {
            if (row_in[j] > max) {
               max = row_in[j]; 
            }
        }

        float sum = 0.f;
        std::vector<float> exps(col_size);
        for (size_t j = 0; j < col_size; j++) {
            float e = std::exp(row_in[j] - max);
            exps[j] = e;
            sum += e;
        }

        for (size_t j = 0; j < col_size; j++) {
            row_out[j] = exps[j] / sum;
        }
    }

    return output;
}

int main() {
    constexpr size_t rowCount = 8192;
    constexpr size_t rowSize = 16384;
    for (size_t i = 0; i < row_size * col_size; i++) {
        input[i] = ((float)rand()/RAND_MAX)*20.f - 10.f;
    }


    auto resRef = SofrowCountRef(input, rowCount);
    auto res = SofrowCountCUDA(input, rowCount);
    float error = 0.0f;
    for (size_t i = 0; i < rowCount * rowSize; ++i) {
        error = std::max(std::fabs(res[i] - resRef[i]), error);
    }
    std::cout << "Max error: " << error / maxVal << std::endl;

    std::vector<double> time_list;
    for (int i = 0; i < 4; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        SoftmaxCUDA(input, rowCount);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;
        time_list.push_back(duration.count());
    }
    double time = *std::min_element(time_list.begin(), time_list.end());
    std::cout << "Time: " << time << " seconds" << std::endl;
}
#endif