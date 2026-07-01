#include "softmax_cuda.h"


__global__ void row_max_kernel(const float* input, float* output, int rows, int cols) {
    int row_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (row_index >= rows) {
        return;
    }

    float row_max = -INFINITY;
    float x = 0.f;
    for (size_t j = 0; j < cols; ++j) {
        x = input[row_index * cols + j];
        if (x > row_max) {
            row_max = x;
        }
    }
    output[row_index] = row_max;
}


__global__ void exp_shift_kernel(float* input, float* row_max, int rows, int cols) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < rows && j < cols) {
        input[i * cols + j] = expf(input[i * cols + j] - row_max[i]);
    }
}


__global__ void row_inv_sum_kernel(float* input, float* row_sum, int rows, int cols) {
    int row_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (row_index >= rows) {
        return;
    }

    float sum = 0.f;
    for (size_t j = 0; j < cols; ++j) {
        sum += input[row_index * cols + j];
    }
    row_sum[row_index] = 1.0f / sum;
}

__global__ void softmax_kernel(float* input, float* row_inv_sum, int rows, int cols) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < rows && j < cols) {
        input[i * cols + j] *= row_inv_sum[i];
    }
}

std::vector<float> SoftmaxCUDA(const std::vector<float>& input, int row_count) {
    int input_size = input.size();
    int cols = input_size / row_count;

    std::vector<float> result_cpu(input_size, 0.f);

    float* gpu_input = nullptr;
    float* gpu_row_var = nullptr;

    cudaMalloc(&gpu_input, input_size * sizeof(float));
    cudaMalloc(&gpu_row_var, row_count * sizeof(float));

    cudaMemcpy(gpu_input, input.data(), input_size * sizeof(float), cudaMemcpyHostToDevice);


    const int block_size = 256;
    const int num_blocks = (row_count + block_size - 1) / block_size;
    row_max_kernel<<<num_blocks, block_size>>>(gpu_input, gpu_row_var, row_count, cols);
    cudaDeviceSynchronize();

    dim3 threads_per_block(16, 16);
    dim3 blocks(
        (cols + threads_per_block.x - 1) / threads_per_block.x,
        (row_count + threads_per_block.y - 1) / threads_per_block.y
    );
    exp_shift_kernel<<<blocks, threads_per_block>>>(gpu_input, gpu_row_var, row_count, cols);
    cudaDeviceSynchronize();

    row_inv_sum_kernel<<<num_blocks, block_size>>>(gpu_input, gpu_row_var, row_count, cols);
    cudaDeviceSynchronize();

    softmax_kernel<<<blocks, threads_per_block>>>(gpu_input, gpu_row_var, row_count, cols);

    cudaMemcpy(result_cpu.data(), gpu_input, input_size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(gpu_input);
    cudaFree(gpu_row_var);

    return result_cpu;
}
