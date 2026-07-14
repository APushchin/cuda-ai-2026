#include "softmax_cuda.h"

#include <cmath>
#include <iostream>
#include <random>
#include <chrono>
#include <cfloat>
#include <vector>
#include <algorithm>

#define WARP_SIZE 32
#define BLOCK_SIZE 256

__global__ void SoftmaxCUDAKernel(float* input, int rowCount, int rowSize) {
    int rowIdx = blockIdx.x;
    if (rowIdx >= rowCount) {
        return;
    }

    int tIdx = threadIdx.x;
    int warp_id = tIdx / WARP_SIZE;
    int lane_id = tIdx % WARP_SIZE;
    
    float* row_inputData = input + rowIdx * rowSize;

    float t_max = -FLT_MAX;
    for (int col = tIdx; col < rowSize; col += blockDim.x) {
        t_max = fmaxf(t_max, row_inputData[col]);
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
        float exp_val = expf(row_inputData[col] - global_max);
        row_inputData[col] = exp_val;
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
        row_inputData[col] *= k;
    }
}

std::vector<float> SoftmaxCUDA(const std::vector<float>& input, int rowCount) {
    const int dataSize = input.size();
        std::vector<float> output(dataSize); 

    const int rowSize = dataSize / rowCount;
    const float* inputData = input.data();

    float* devInput = nullptr;
    cudaMalloc(&devInput, dataSize * sizeof(float));

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaMemcpyAsync(devInput, inputData, dataSize * sizeof(float), cudaMemcpyHostToDevice, stream);

    SoftmaxCUDAKernel<<<rowCount, BLOCK_SIZE, 0, stream>>>(devInput, rowCount, rowSize);
    cudaMemcpyAsync(output.data(), devInput, dataSize * sizeof(float), cudaMemcpyDeviceToHost, stream);

    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
    cudaFree(devInput);
    
    return output;
}

#if 0
std::vector<float> SoftmaxRef(const std::vector<float>& input, int rowCount) {
    size_t rowSize = input.size() / rowCount;
    std::vector<float> output(rowCount * rowSize);

    const float* inptr = input.data();
    float* outptr = output.data();

    for (size_t i = 0; i < rowCount; i++) {
        const float* row_in = inptr + i * rowSize;
        float* row_out = outptr + i * rowSize;

        float max_val = std::numeric_limits<float>::lowest();
        for (size_t j = 0; j < rowSize; j++) {
            if (row_in[j] > max_val) {
               max_val = row_in[j]; 
            }
        }

        float sum = 0.f;
        std::vector<float> exps(rowSize);
        for (size_t j = 0; j < rowSize; j++) {
            float e = std::exp(row_in[j] - max_val);
            exps[j] = e;
            sum += e;
        }

        for (size_t j = 0; j < rowSize; j++) {
            row_out[j] = exps[j] / sum;
        }
    }

    return output;
}

int main() {
    // FIXED: Clear and consistent naming
    size_t rowCount = 8192;
    size_t rowSize = 16384;
    
    std::vector<float> input(rowCount * rowSize);
    for (size_t i = 0; i < rowCount * rowSize; i++) {
        input[i] = ((float)rand() / RAND_MAX) * 20.f - 10.f;
    }

    // Warming-up
    auto output = SoftmaxCUDA(input, rowCount);

    std::vector<float> outref = SoftmaxRef(input, rowCount);
    
    float error = 0.f;
    for (size_t i = 0; i < rowCount * rowSize; i++) {
        error = std::max(error, std::abs(output[i] - outref[i]));
    }
    std::cout << "Absolute max error: " << error << std::endl;
    
    // Performance Measuring
    int nIters = 10;
    double min_t = 0.0;

    for (int i = 0; i < nIters; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        auto perf_output = SoftmaxCUDA(input, rowCount);
        std::chrono::duration<double> duration = std::chrono::high_resolution_clock::now() - start;
        double t = duration.count();
        min_t = (i == 0) ? t : std::min(min_t, t);
    }

    std::cout << "Min execution time: \t" << min_t << " seconds" << std::endl;

    return 0;
}
#endif