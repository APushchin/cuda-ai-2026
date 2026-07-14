#include <vector>
#include <stdexcept>

#include <cuda_runtime.h>

#include "naive_gemm_cuda.h"

#define DIV_UP(a, b) ( ((a) + (b) - 1) / (b) )

__global__ void naive_square_matrix_gemm_kernel(
    const float *input_lhs,
    const float *input_rhs,
    float *result,
    int shape_length
) {
    const auto column_idx = threadIdx.x + blockIdx.x * blockDim.x;
    const auto row_idx = threadIdx.y + blockIdx.y * blockDim.y;
    if (row_idx >= shape_length || column_idx >= shape_length) return;

    float sum = 0;
    for (int i = 0; i < shape_length; ++i) {
        sum += (
            input_lhs[row_idx * shape_length + i] *
            input_rhs[i * shape_length + column_idx]
        );
    }
    result[row_idx * shape_length + column_idx] = sum;
}

std::vector<float> NaiveGemmCUDA(
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

    cudaStream_t stream;
    error = cudaStreamCreate(&stream);
    if (error != cudaSuccess) {
        cudaFree(device_input_lhs);
        cudaFree(device_input_rhs);
        cudaFree(device_result);
        throw std::runtime_error("Failed to create CUDA stream");
    }

    const auto input_lhs_raw_data = (void*)(a.data());
    error = cudaHostRegister(input_lhs_raw_data, matrix_size, cudaHostRegisterDefault);
    if (error != cudaSuccess) {
        cudaFree(device_input_lhs);
        cudaFree(device_input_rhs);
        cudaFree(device_result);
        cudaStreamDestroy(stream);
        throw std::runtime_error("Failed to register host memory for input A");
    }

    const auto input_rhs_raw_data = (void*)(b.data());
    error = cudaHostRegister(input_rhs_raw_data, matrix_size, cudaHostRegisterDefault);
    if (error != cudaSuccess) {
        cudaFree(device_input_lhs);
        cudaFree(device_input_rhs);
        cudaFree(device_result);
        cudaHostUnregister(input_lhs_raw_data);
        cudaStreamDestroy(stream);
        throw std::runtime_error("Failed to register host memory for input A");
    }

    std::vector<float> result;
    result.reserve(elements_number);
    const auto result_raw_data = (void*)(result.data());
    error = cudaHostRegister(result_raw_data, matrix_size, cudaHostRegisterDefault);
    if (error != cudaSuccess) {
        cudaFree(device_input_lhs);
        cudaFree(device_input_rhs);
        cudaFree(device_result);
        cudaHostUnregister(input_lhs_raw_data);
        cudaHostUnregister(input_rhs_raw_data);
        cudaStreamDestroy(stream);
        throw std::runtime_error("Failed to register host memory for input");
    }

    error = cudaMemcpyAsync(
        device_input_lhs,
        input_lhs_raw_data,
        matrix_size,
        cudaMemcpyHostToDevice,
        stream
    );
    if (error != cudaSuccess) {
        cudaFree(device_input_lhs);
        cudaFree(device_input_rhs);
        cudaFree(device_result);
        cudaHostUnregister(input_lhs_raw_data);
        cudaHostUnregister(input_rhs_raw_data);
        cudaHostUnregister(result_raw_data);
        cudaStreamDestroy(stream);
        throw std::runtime_error("Failed to copy input A to device");
    }

    error = cudaMemcpyAsync(
        device_input_rhs,
        input_rhs_raw_data,
        matrix_size,
        cudaMemcpyHostToDevice,
        stream
    );
    if (error != cudaSuccess) {
        cudaFree(device_input_lhs);
        cudaFree(device_input_rhs);
        cudaFree(device_result);
        cudaHostUnregister(input_lhs_raw_data);
        cudaHostUnregister(input_rhs_raw_data);
        cudaHostUnregister(result_raw_data);
        cudaStreamDestroy(stream);
        throw std::runtime_error("Failed to copy input A to device");
    }

    constexpr auto square_block_size = 16;
    const auto threads_per_block = dim3(square_block_size, square_block_size);
    const auto block_count = dim3(
        DIV_UP(n, threads_per_block.x),
        DIV_UP(n, threads_per_block.y)
    );
    naive_square_matrix_gemm_kernel<<<block_count, threads_per_block>>>(
        device_input_lhs, device_input_rhs, device_result, n
    );

    error = cudaMemcpyAsync(
        result_raw_data,
        device_result,
        matrix_size,
        cudaMemcpyDeviceToHost,
        stream
    );
    if (error != cudaSuccess) {
        cudaFree(device_input_lhs);
        cudaFree(device_input_rhs);
        cudaFree(device_result);
        cudaHostUnregister(input_lhs_raw_data);
        cudaHostUnregister(input_rhs_raw_data);
        cudaHostUnregister(result_raw_data);
        cudaStreamDestroy(stream);
        throw std::runtime_error("Failed to copy result to host");
    }

    error = cudaStreamSynchronize(stream);
    if (error != cudaSuccess) {
        cudaFree(device_input_lhs);
        cudaFree(device_input_rhs);
        cudaFree(device_result);
        cudaHostUnregister(input_lhs_raw_data);
        cudaHostUnregister(input_rhs_raw_data);
        cudaHostUnregister(result_raw_data);
        cudaStreamDestroy(stream);
        throw std::runtime_error("Failed to synchronize CUDA stream");
    }

    cudaFree(device_input_lhs);
    cudaFree(device_input_rhs);
    cudaFree(device_result);
    cudaHostUnregister(input_lhs_raw_data);
    cudaHostUnregister(input_rhs_raw_data);
    cudaHostUnregister(result_raw_data);
    cudaStreamDestroy(stream);

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
    NaiveGemmCUDA(input_lhs, input_rhs, MATRIX_SHAPE); // warming up

    float max_absolute_error = 0.f;
    float max_relative_error = 0.f;

    std::vector<double> time_list;
    for (int experiment_id = 0; experiment_id < NUM_EXPERIMENTS; ++experiment_id) {
        const auto start = std::chrono::high_resolution_clock::now();
        const auto result = NaiveGemmCUDA(input_lhs, input_rhs, MATRIX_SHAPE);
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
                std::cout << "Found relative_error=" << relative_error
                          << ", more than threshold=" << RELATIVE_ERROR_THRESHOLD
                          << ", result=" << result[j] << " but reference=" << result_reference[j]
                          << std::endl;
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
