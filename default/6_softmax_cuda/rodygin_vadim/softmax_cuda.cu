// softmax_cuda.cu
#include "softmax_cuda.h"
#include <cuda_runtime.h>
#include <cmath>
#include <stdexcept>

__global__ void FindRowMaxKernel(const float* input, float* row_maxes, 
                                  int total_size, int col_size, int num_rows) {
    extern __shared__ float shared_max[];
    
    int row = blockIdx.x;
    if (row >= num_rows) return;
    
    int col_idx = threadIdx.x;
    int idx = row * col_size + col_idx;
    
    if (col_idx < col_size && idx < total_size) {
        shared_max[col_idx] = input[idx];
    } else {
        shared_max[col_idx] = -INFINITY;
    }
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (col_idx < stride) {
            shared_max[col_idx] = fmaxf(shared_max[col_idx], 
                                         shared_max[col_idx + stride]);
        }
        __syncthreads();
    }
    
    if (threadIdx.x == 0) {
        row_maxes[row] = shared_max[0];
    }
}

__global__ void SoftmaxKernel(const float* input, const float* row_maxes, 
                               float* output, int total_size, 
                               int col_size, int num_rows) {
    extern __shared__ float shared_data[];
    
    int row = blockIdx.x;
    if (row >= num_rows) return;
    
    int col_idx = threadIdx.x;
    int idx = row * col_size + col_idx;
    
    float row_max = row_maxes[row];
    
    float exp_val = 0.0f;
    if (col_idx < col_size && idx < total_size) {
        exp_val = expf(input[idx] - row_max);
    }
    
    shared_data[col_idx] = exp_val;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (col_idx < stride) {
            shared_data[col_idx] += shared_data[col_idx + stride];
        }
        __syncthreads();
    }
    
    float sum_exp = shared_data[0];
    
    if (col_idx < col_size && idx < total_size) {
        output[idx] = exp_val / sum_exp;
    }
}

std::vector<float> SoftmaxCUDA(const std::vector<float>& input, int row_size) {
    if (input.empty()) {
        return input;
    }
    
    if (row_size <= 0) {
        throw std::invalid_argument("row_size must be positive");
    }
    
    size_t total_size = input.size();
    
    if (total_size % row_size != 0) {
        throw std::invalid_argument("input size must be divisible by row_size");
    }
    
    int num_rows = row_size;  // parameter is the number of rows
    int col_size = static_cast<int>(total_size) / row_size;  // elements per row
    
    float *d_input, *d_row_maxes, *d_output;
    
    cudaMalloc(&d_input, total_size * sizeof(float));
    cudaMalloc(&d_row_maxes, num_rows * sizeof(float));
    cudaMalloc(&d_output, total_size * sizeof(float));
    
    cudaMemcpy(d_input, input.data(), total_size * sizeof(float), cudaMemcpyHostToDevice);
    
    int threads_per_block = 256;
    int blocks_per_grid = num_rows;
    size_t shared_mem_size = threads_per_block * sizeof(float);
    
    FindRowMaxKernel<<<blocks_per_grid, threads_per_block, shared_mem_size>>>(
        d_input, d_row_maxes, static_cast<int>(total_size), col_size, num_rows);
    
    SoftmaxKernel<<<blocks_per_grid, threads_per_block, shared_mem_size>>>(
        d_input, d_row_maxes, d_output, static_cast<int>(total_size), col_size, num_rows);
    
    cudaDeviceSynchronize();
    
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        cudaFree(d_input);
        cudaFree(d_row_maxes);
        cudaFree(d_output);
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(error));
    }
    
    std::vector<float> output(total_size);
    cudaMemcpy(output.data(), d_output, total_size * sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaFree(d_input);
    cudaFree(d_row_maxes);
    cudaFree(d_output);
    
    return output;
}