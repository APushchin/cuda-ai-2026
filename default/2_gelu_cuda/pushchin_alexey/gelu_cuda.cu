#include <vector>
#include <stdexcept>
#include <cstring>
#include <cmath>

#include <cuda_runtime.h>

#include "gelu_cuda.h"

#define DIV_UP(a, b) ( ((a) + (b) - 1) / (b) )

__global__ void gelu_kernel(
    const float* __restrict__ input,
    float*       __restrict__ output,
    size_t       length
) {
    const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= length) return;

    const auto x = input[idx];

    constexpr auto SQRT_2_OVER_PI = 0.7978845608f; // std::sqrt(2.0f / M_PI)
    constexpr auto COEFFICIENT = 0.044715f;

    const auto tanh_argument = SQRT_2_OVER_PI * (x + COEFFICIENT * x * x * x);

    output[idx] = 0.5f * x * (1.f + tanhf(tanh_argument));
}

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    const auto length = input.size();
    const auto size_in_bytes = length * sizeof(float);

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

    auto output = std::vector<float>(length);
    const auto result_raw_data = (void*)(result.data());
    error = cudaHostRegister(result_raw_data, size_in_bytes, cudaHostRegisterDefault);
    if (error != cudaSuccess) {
        cudaFree(device_input_buffer);
        cudaFree(device_output_buffer);
        cudaStreamDestroy(stream);
        throw std::runtime_error("Failed to register host memory for input");
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
        cudaStreamDestroy(stream);
        throw std::runtime_error("Failed to copy data to device");
    }

    constexpr int threads_per_block = 256;
    const auto num_blocks = DIV_UP(length, threads_per_block);
    gelu_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
        device_input_buffer,
        device_output_buffer,
        length
    );

    error = cudaMemcpyAsync(
        result_raw_data,
        device_output_buffer,
        size_in_bytes,
        cudaMemcpyDeviceToHost,
        stream
    );
    if (error != cudaSuccess) {
        cudaFree(device_input_buffer);
        cudaFree(device_output_buffer);
        cudaHostUnregister(input_raw_data);
        cudaHostUnregister(result_raw_data);
        cudaStreamDestroy(stream);
        throw std::runtime_error("Failed to copy output from device");
    }

    error = cudaStreamSynchronize(stream);
    if (error != cudaSuccess) {
        cudaFree(device_input_buffer);
        cudaFree(device_output_buffer);
        cudaHostUnregister(input_raw_data);
        cudaHostUnregister(result_raw_data);
        cudaStreamDestroy(stream);
        throw std::runtime_error("Failed to synchronize CUDA stream");
    }

    cudaFree(device_input_buffer);
    cudaFree(device_output_buffer);
    cudaHostUnregister(input_raw_data);
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

std::vector<float> GeluReference(const std::vector<float>& input) {
    auto output = std::vector<float>(input.size());
    std::transform(
        input.begin(),
        input.end(),
        std::back_inserter(output),
        [](float x){
            return 0.5f * x * (
                1.0f + std::tanh(
                    std::sqrt(2.0f / M_PI) * (x + 0.044715f * x*x*x)
                )
            );
        }
    );
    return output;
}

constexpr size_t INPUT_LENGTH = 10000000;
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
    const auto input = generate_input(INPUT_LENGTH);
    const auto result_reference = GeluReference(input);
    GeluCUDA(input); // warming up

    float max_absolute_error = 0.f;
    float max_relative_error = 0.f;

    std::vector<double> time_list;
    for (int experiment_id = 0; experiment_id < NUM_EXPERIMENTS; ++experiment_id) {
        const auto start = std::chrono::high_resolution_clock::now();
        const auto result = GeluCUDA(input);
        const auto finish = std::chrono::high_resolution_clock::now();
        const auto duration = std::chrono::duration<double>(finish - start);
        time_list.push_back(duration.count());

        for (int j = 0; j < INPUT_LENGTH; ++j) {
            max_absolute_error = std::max(
                std::abs(result[j] - result_reference[j]),
                max_absolute_error
            );
            const auto relative_error = std::abs(result[j] / result_reference[j] - 1.f);
            if (relative_error >= 0.001f) {
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
    }
    const auto total_time = std::accumulate(time_list.begin(), time_list.end(), 0.f);
    const auto min_time = *std::min_element(time_list.begin(), time_list.end());

    std::cout << "Processing " << INPUT_LENGTH << " numbers " << NUM_EXPERIMENTS << " times "
              << "took " << total_time << " seconds"
              << ": mean=" << total_time / NUM_EXPERIMENTS << 's'
              << ", min=" << min_time << 's' << std::endl
              << "Max errors: absolute=" << max_absolute_error
              << ", relative=" << max_relative_error << std::endl;

}

#endif // DEBUG
