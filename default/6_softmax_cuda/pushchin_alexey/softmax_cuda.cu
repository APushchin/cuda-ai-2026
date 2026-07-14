#include <cfloat> // FLT_MAX
#include <stdexcept>
#include <vector>

#include <iostream>  // std::cout

#include "softmax_cuda.h"

#define BLOCK_SIZE 32

__global__ void softmax_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    int row_count,
    int row_length
) {
    extern __shared__ float shared_memory[];

    const auto row_idx = blockIdx.x;
    const auto thread_idx = threadIdx.x;

    auto local_max = -FLT_MAX;
    for (int i = thread_idx; i < row_length; i += BLOCK_SIZE) {
        local_max = fmaxf(local_max, input[row_idx * row_length + i]);
    }

    auto* current_shared_memory = shared_memory + BLOCK_SIZE;
    current_shared_memory[thread_idx] = local_max;
    __syncthreads();

    __shared__ float row_max;
    if(thread_idx == 0) {
        row_max = -FLT_MAX;
        for (int i = 0; i < BLOCK_SIZE; ++i) {
            row_max = fmaxf(current_shared_memory[i], row_max);
        }
    }

    auto sum = 0.f;
    for (int i = thread_idx; i < row_length; i += BLOCK_SIZE) {
        sum += expf(input[row_idx * row_length + i] - row_max);
    }
    current_shared_memory[thread_idx] = sum;
    __syncthreads();

    __shared__ float row_sum;
    if(thread_idx == 0) {
        row_sum = 0.f;
        for(int i = 0; i < BLOCK_SIZE; ++i) {
            row_sum += current_shared_memory[i];
        }
    }
    __syncthreads();

    for (int i = thread_idx; i < row_length; i += BLOCK_SIZE) {
        output[row_idx * row_length + i] = (
            expf(input[row_idx * row_length + i] - row_max) / row_sum
        );
    }
}

std::vector<float> SoftmaxCUDA(const std::vector<float>& input, int row_count)
{
    const auto elements_number = input.size();
    const auto size_in_bytes = elements_number * sizeof(float);
    const auto row_length = elements_number / row_count;

    float* device_input_buffer  = nullptr;
    auto error = cudaMalloc(&device_input_buffer, size_in_bytes);
    if (error != cudaSuccess) {
        throw std::runtime_error("Failed to allocate device memory for input");
    }

    float* device_output_buffer = nullptr;
    error = cudaMalloc(&device_output_buffer, size_in_bytes);
    if (error != cudaSuccess) {
        cudaFree(device_input_buffer);
        throw std::runtime_error("Failed to allocate device memory for output");
    }

    cudaStream_t stream;
    error = cudaStreamCreate(&stream);
    if (error != cudaSuccess) {
        cudaFree(device_input_buffer);
        cudaFree(device_output_buffer);
        throw std::runtime_error("Failed to create CUDA stream");
    }

    const auto input_raw_data = (void*)(input.data());
    error = cudaHostRegister(input_raw_data, size_in_bytes, cudaHostRegisterDefault);
    if (error != cudaSuccess) {
        cudaFree(device_input_buffer);
        cudaFree(device_output_buffer);
        cudaStreamDestroy(stream);
        throw std::runtime_error("Failed to register host memory for input");
    }

    std::vector<float> result;
    result.reserve(elements_number);
    const auto output_raw_data = (void*)(result.data());
    error = cudaHostRegister(output_raw_data, size_in_bytes, cudaHostRegisterDefault);
    if (error != cudaSuccess) {
        cudaFree(device_input_buffer);
        cudaFree(device_output_buffer);
        cudaHostUnregister(input_raw_data);
        cudaStreamDestroy(stream);
        throw std::runtime_error("Failed to register host memory for output");
    }

    error = cudaMemcpyAsync(
        device_input_buffer,
        input_raw_data,
        size_in_bytes,
        cudaMemcpyHostToDevice,
        stream
    );
    if (error != cudaSuccess) {
        cudaFree(device_input_buffer);
        cudaFree(device_output_buffer);
        cudaHostUnregister(input_raw_data);
        cudaHostUnregister(output_raw_data);
        cudaStreamDestroy(stream);
        throw std::runtime_error("Failed to copy data to device");
    }

    const auto shared_memory_size = (row_length + BLOCK_SIZE) * sizeof(float);
    softmax_kernel<<<dim3(row_count), dim3(BLOCK_SIZE), shared_memory_size, stream>>>(
        device_output_buffer, device_input_buffer, row_count, row_length
    );

    error = cudaMemcpyAsync(
        output_raw_data,
        device_output_buffer,
        size_in_bytes,
        cudaMemcpyDeviceToHost,
        stream
    );
    if (error != cudaSuccess) {
        cudaFree(device_input_buffer);
        cudaFree(device_output_buffer);
        cudaHostUnregister(input_raw_data);
        cudaHostUnregister(output_raw_data);
        cudaStreamDestroy(stream);
        throw std::runtime_error("Failed to copy output from device");
    }

    error = cudaStreamSynchronize(stream);
    if (error != cudaSuccess) {
        cudaFree(device_input_buffer);
        cudaFree(device_output_buffer);
        cudaHostUnregister(input_raw_data);
        cudaHostUnregister(output_raw_data);
        cudaStreamDestroy(stream);
        throw std::runtime_error("Failed to synchronize CUDA stream");
    }

    cudaFree(device_input_buffer);
    cudaFree(device_output_buffer);
    cudaHostUnregister(input_raw_data);
    cudaHostUnregister(output_raw_data);
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
constexpr size_t ROW_LENGTH = 1024;
constexpr size_t ROW_COUNT = 1024;
constexpr size_t NUM_EXPERIMENTS = 5;

std::vector<float> SoftmaxReference(const std::vector<float>& input, int row_count) {
    const auto elements_number = input.size();
    const auto row_length = elements_number / row_count;
    std::vector<float> output(elements_number);
    for (int i = 0; i < row_count; ++i) {
        const auto max_value = *std::max_element(
            input.begin() + i * row_length,
            input.begin() + i * (row_length + 1)
        );
        auto sum = 0.f;
        auto exponents = std::vector<float>();
        exponents.reserve(row_length);
        for (int i = 0; i < row_length; ++i) {
            exponents[i] = std::exp(input[i * row_length + i] - max_value);
            sum += exponents[i];
        }
        auto coefficient = 1.f / sum;
        for (int i = 0; i < row_length; ++i) {
            output[i * row_length + i] = exponents[i] * coefficient;
        }
    }
    return output;
}

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
    const auto elements_number = ROW_LENGTH * ROW_COUNT;
    const auto input = generate_input(elements_number);
    const auto result_reference = SoftmaxReference(input, ROW_COUNT);
    SoftmaxCUDA(input, ROW_COUNT); // warming up

    float max_absolute_error = 0.f;
    float max_relative_error = 0.f;

    std::vector<double> time_list;
    for (int experiment_id = 0; experiment_id < NUM_EXPERIMENTS; ++experiment_id) {
        const auto start = std::chrono::high_resolution_clock::now();
        const auto result = SoftmaxCUDA(input, ROW_COUNT);
        const auto finish = std::chrono::high_resolution_clock::now();
        const auto duration = std::chrono::duration<double>(finish - start);
        time_list.push_back(duration.count());

        int num_errors_found = 0;
        for (int i = 0; i < elements_number; ++i) {
            max_absolute_error = std::max(
                std::abs(result[i] - result_reference[i]),
                max_absolute_error
            );
            const auto relative_error = std::abs(result[i] / result_reference[i] - 1.f);
            if (relative_error >= 0.001f) {
                ++num_errors_found;
                // std::cout << "Found relative_error=" << relative_error
                //           << ", more than threshold=" << RELATIVE_ERROR_THRESHOLD
                //           << ", result=" << result[i] << " but reference=" << result_reference[i]
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

    std::cout << "Taking Softmax from " << ROW_LENGTH << 'x' << ROW_COUNT << " matrix "
              << NUM_EXPERIMENTS << " times "
              << "took " << total_time << " seconds"
              << ": mean=" << total_time / NUM_EXPERIMENTS << 's'
              << ", min=" << min_time << 's' << std::endl
              << "Max errors: absolute=" << max_absolute_error
              << ", relative=" << max_relative_error << std::endl;

}

#endif // DEBUG
