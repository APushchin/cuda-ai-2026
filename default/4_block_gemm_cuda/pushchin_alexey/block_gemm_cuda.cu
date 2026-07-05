#include <vector>
#include <stdexcept>

#include <cuda_runtime.h>

#include "block_gemm_cuda.h"

constexpr auto square_block_size = 16;

#define DIV_UP(device_input_lhs, device_input_rhs) ( ((device_input_lhs) + (device_input_rhs) - 1) / (device_input_rhs) )

__global__ void block_square_matrix_gemm_kernel(
    const float *input_lhs,
    const float *input_rhs,
    float *result,
    int shape_length
) {
    const auto thread_column_idx = threadIdx.x;
    const auto thread_row_idx = threadIdx.y;

    const auto column_idx = blockIdx.x * square_block_size + thread_column_idx;
    const auto row_idx = blockIdx.y * square_block_size + thread_row_idx;
    if (row_idx >= shape_length || column_idx >= shape_length) return;

    __shared__ float shared_input_lhs[square_block_size][square_block_size];
    __shared__ float shared_input_rhs[square_block_size][square_block_size];
    float sum = 0.f;
    for (int block_idx = 0; block_idx < DIV_UP(shape_length, square_block_size); ++block_idx) {
        const auto block_column_idx = block_idx * square_block_size + thread_column_idx;
        const auto block_row_idx = block_idx * square_block_size + thread_row_idx;

        shared_input_lhs[thread_row_idx][thread_column_idx] =
            (row_idx < shape_length && block_column_idx < shape_length) ?
            input_lhs[row_idx * shape_length + block_column_idx] :
            0.0f;
        shared_input_rhs[thread_row_idx][thread_column_idx] =
            (column_idx < shape_length && block_row_idx < shape_length) ?
            input_rhs[block_row_idx * shape_length + column_idx] :
            0.0f;
        __syncthreads();

        for (int i = 0; i < square_block_size; ++i) {
            sum += shared_input_lhs[thread_row_idx][i] * shared_input_rhs[i][thread_column_idx];
        }
        __syncthreads();
    }

    result[row_idx * shape_length + column_idx] = sum;
}

std::vector<float> BlockGemmCUDA(
    const std::vector<float>& a,
    const std::vector<float>& b,
    int n
) {
    const auto elements_number = n * n;
    const auto matrix_size = elements_number * sizeof(float);

    float* device_input_lhs = nullptr;
    cudaError_t error = cudaMalloc(&device_input_lhs, matrix_size);
    if (error != cudaSuccess) {
        throw std::runtime_error("Failed to allocate device memory for input A");
    }

    float* device_input_rhs = nullptr;
    error = cudaMalloc(&device_input_rhs, matrix_size);
    if (error != cudaSuccess) {
        cudaFree(device_input_lhs);
        throw std::runtime_error("Failed to allocate device memory for input B");
    }

    float* device_result = nullptr;
    error = cudaMalloc(&device_result, matrix_size);
    if (error != cudaSuccess) {
        cudaFree(device_input_lhs);
        cudaFree(device_input_rhs);
        throw std::runtime_error("Failed to allocate device memory for output");
    }

    error = cudaMemcpy(device_input_lhs, a.data(), matrix_size, cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        cudaFree(device_input_lhs);
        cudaFree(device_input_rhs);
        cudaFree(device_result);
        throw std::runtime_error("Failed to copy input A to device");
    }

    error = cudaMemcpy(device_input_rhs, b.data(), matrix_size, cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        cudaFree(device_input_lhs);
        cudaFree(device_input_rhs);
        cudaFree(device_result);
        throw std::runtime_error("Failed to copy input B to device");
    }

    const auto threads_per_block = dim3(square_block_size, square_block_size);
    const auto blocks = dim3(
        DIV_UP(n, threads_per_block.x),
        DIV_UP(n, threads_per_block.y)
    );
    block_square_matrix_gemm_kernel<<<blocks, threads_per_block>>>(
        device_input_lhs, device_input_rhs, device_result, n
    );

    auto result = std::vector<float>(elements_number);
    error = cudaMemcpy(result.data(), device_result, matrix_size, cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) {
        cudaFree(device_input_lhs);
        cudaFree(device_input_rhs);
        cudaFree(device_result);
        throw std::runtime_error("Failed to copy result to host");
    }

    cudaFree(device_input_lhs);
    cudaFree(device_input_rhs);
    cudaFree(device_result);

    return result;
}


#define DEBUG false

#if DEBUG

#include <random>    // std::mt19937, std::uniform_real_distribution
#include <iostream>  // std::cout
#include <chrono>    // std::chrono
#include <algorithm> // std::generate, std::transform, std::min_element, std::accumulate

#define RELATIVE_ERROR_THRESHOLD 0.001f

static std::vector<float> GemmReference(
    const std::vector<float>& a,
    const std::vector<float>& b,
    int n
) {
    auto result = std::vector<float>(n * n, 0.0f);
    for (int row_idx = 0; row_idx < n; ++row_idx) {
        for (int column_idx = 0; column_idx < n; ++column_idx) {
            for (int i = 0; i < n; ++i) {
                result[row_idx * n + column_idx] += (
                    a[row_idx * n + i] *
                    b[i * n + column_idx]
                );
            }
        }
    }
    return result;
}


constexpr size_t MATRIX_SHAPE = 1024;
constexpr size_t NUM_EXPERIMENTS = 5;

const std::vector<float> generate_input(size_t size) {
    std::vector<float> random_floats(size);

    std::random_device random_device;
    std::mt19937 generator(random_device());
    std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
    std::generate(
        random_floats.begin(),
        random_floats.end(),
        [&]() {return distribution(generator);}
    );

    return random_floats;
}

int main() {
    const auto elements_number = MATRIX_SHAPE * MATRIX_SHAPE;
    const auto input_lhs = generate_input(elements_number);
    const auto input_rhs = generate_input(elements_number);
    const auto result_reference = GemmReference(input_lhs, input_rhs, MATRIX_SHAPE);
    BlockGemmCUDA(input_lhs, input_rhs, MATRIX_SHAPE); // warming up

    float max_absolute_error = 0.f;
    float max_relative_error = 0.f;

    std::vector<double> time_list;
    for (int experiment_id = 0; experiment_id < NUM_EXPERIMENTS; ++experiment_id) {
        const auto start = std::chrono::high_resolution_clock::now();
        const auto result = BlockGemmCUDA(input_lhs, input_rhs, MATRIX_SHAPE);
        const auto finish = std::chrono::high_resolution_clock::now();
        const auto duration = std::chrono::duration<double>(finish - start);
        time_list.push_back(duration.count());

        int num_errors_found = 0;
        for (int j = 0; j < elements_number; ++j) {
            max_absolute_error = std::max(
                std::abs(result[j] - result_reference[j]),
                max_absolute_error
            );
            const auto relative_error = std::abs(result[j] / result_reference[j] - 1.f);
            if (relative_error >= 0.001f) {
                ++num_errors_found;
                // std::cout << "Found relative_error=" << relative_error
                //           << ", more than threshold=" << RELATIVE_ERROR_THRESHOLD
                //           << ", result=" << result[j] << " but reference=" << result_reference[j]
                //           << std::endl;
            }
            max_relative_error = std::max(
                relative_error,
                max_relative_error
            );
        }
        std::cout << "Found " << num_errors_found << " errors"
                  << " (" << num_errors_found / float(elements_number) * 100 << "%)" << std::endl;
    }
    const auto total_time = std::accumulate(time_list.begin(), time_list.end(), 0.f);
    const auto min_time = *std::min_element(time_list.begin(), time_list.end());

    std::cout << "Multiplying " << MATRIX_SHAPE << 'x' << MATRIX_SHAPE << " matrices "
              << NUM_EXPERIMENTS << " times "
              << "took " << total_time << " seconds"
              << ": mean=" << total_time / NUM_EXPERIMENTS << 's'
              << ", min=" << min_time << 's' << std::endl
              << "Max errors: absolute=" << max_absolute_error
              << ", relative=" << max_relative_error << std::endl;

}

#endif // DEBUG
