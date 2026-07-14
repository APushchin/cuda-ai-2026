#include "gelu_omp.h"

std::vector<float> GeluOMP(const std::vector<float>& input) {
    // Processing 10000000 numbers 5 times took 0.123938 seconds: mean=0.0247876s, min=0.0153677s
    std::vector<float> output(input.size());

    constexpr auto SQRT_2_OVER_PI = 0.7978845608f; // std::sqrt(2.0f / M_PI)
    constexpr auto COEFFICIENT = 0.044715f;
    #pragma omp parallel for simd
    for (int i = 0; i < input.size(); ++i) {
        const auto x = input[i];
        const auto tanh_argument = SQRT_2_OVER_PI * (x + COEFFICIENT * x * x * x);
        float tanh_result;
        if (tanh_argument < -20.f) {
            tanh_result = -1.f;
        } else if (tanh_argument > 20.f) {
            tanh_result = 1.f;
        } else {
            const auto exp_positive = std::exp(tanh_argument);
            const auto exp_negative = std::exp(-tanh_argument);
            tanh_result = (exp_positive - exp_negative) / (exp_positive + exp_negative);
        }
        output[i] = 0.5f * x * (1.0f + tanh_result);
    }

    return output;
}

#define DEBUG false

#if DEBUG

#include <random>   // std::mt19937, std::uniform_real_distribution
#include <iostream> // std::cout
#include <chrono>   // std::chrono

#define RELATIVE_ERROR_THRESHOLD 0.001f

std::vector<float> GeluReference(const std::vector<float>& input) {
    // Processing 10000000 numbers 5 times took 2.34033 seconds: mean=0.468066s, min=0.465243s

    std::vector<float> output;
    output.reserve(input.size());
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

std::vector<float> GeluConstsFactoredOut(const std::vector<float>& input) {
    // Processing 10000000 numbers 5 times took 2.5413 seconds: mean=0.508259s, min=0.504669s

    constexpr float SQRT_2_OVER_PI = 0.7978845608f; // std::sqrt(2.0f / M_PI)
    constexpr float COEFFICIENT = 0.044715f;

    std::vector<float> output;
    output.reserve(input.size());
    std::transform(
        input.begin(),
        input.end(),
        std::back_inserter(output),
        [](float x){
            return 0.5f * x * (1.0f + std::tanh(SQRT_2_OVER_PI * (x + COEFFICIENT * x*x*x)));
        }
    );
    return output;
}

std::vector<float> GeluWithTanhViaExp(const std::vector<float>& input) {
    // Processing 10000000 numbers 5 times took 1.71324 seconds: mean=0.342648s, min=0.341349s

    constexpr float SQRT_2_OVER_PI = 0.7978845608f; // std::sqrt(2.0f / M_PI)
    constexpr float COEFFICIENT = 0.044715f;

    std::vector<float> output;
    output.reserve(input.size());

    std::transform(
        input.begin(),
        input.end(),
        std::back_inserter(output),
        [&](float x){
            const auto tanh_argument = SQRT_2_OVER_PI * (x + COEFFICIENT * x * x * x);
            float tanh_result;
            if (tanh_argument < -20.f) {
                tanh_result = -1.f;
            } else if (tanh_argument > 20.f) {
                tanh_result = 1.f;
            } else {
                const auto exp_positive = std::exp(tanh_argument);
                const auto exp_negative = std::exp(-tanh_argument);
                tanh_result = (exp_positive - exp_negative) / (exp_positive + exp_negative);
            }
            return 0.5f * x * (1.0f + tanh_result);
        }
    );
    return output;
}

std::vector<float> GeluOMP_Basic(const std::vector<float>& input) {
    // Processing 10000000 numbers 5 times took 0.561075 seconds: mean=0.112215s, min=0.109649s
    // Added tweaks: `g++ -O3 -fopenmp -march=native -fopt-info-vec`
    // Processing 10000000 numbers 5 times took 0.137794 seconds: mean=0.0275589s, min=0.0160589s

    std::vector<float> output(input.size());

    constexpr float SQRT_2_OVER_PI = 0.7978845608f; // std::sqrt(2.0f / M_PI)
    constexpr float COEFFICIENT = 0.044715f;
    #pragma omp parallel for
    for (int i = 0; i < input.size(); ++i) {
        const auto x = input[i];
        const auto tanh_argument = SQRT_2_OVER_PI * (x + COEFFICIENT * x * x * x);
        float tanh_result;
        if (tanh_argument < -20.f) {
            tanh_result = -1.f;
        } else if (tanh_argument > 20.f) {
            tanh_result = 1.f;
        } else {
            const auto exp_positive = std::exp(tanh_argument);
            const auto exp_negative = std::exp(-tanh_argument);
            tanh_result = (exp_positive - exp_negative) / (exp_positive + exp_negative);
        }
        output[i] = 0.5f * x * (1.0f + tanh_result);
    }

    return output;
}

std::vector<float> GeluOMP_LoopUnrolling(const std::vector<float>& input) {
    // Processing 10000000 numbers 5 times took 0.185704 seconds: mean=0.0371407s, min=0.0164101s

    std::vector<float> output(input.size());

    constexpr float SQRT_2_OVER_PI = 0.7978845608f; // std::sqrt(2.0f / M_PI)
    constexpr float COEFFICIENT = 0.044715f;
    #pragma omp parallel for
    #pragma omp unroll partial
    for (int i = 0; i < input.size(); ++i) {
        const auto x = input[i];
        const auto tanh_argument = SQRT_2_OVER_PI * (x + COEFFICIENT * x * x * x);
        float tanh_result;
        if (tanh_argument < -20.f) {
            tanh_result = -1.f;
        } else if (tanh_argument > 20.f) {
            tanh_result = 1.f;
        } else {
            const auto exp_positive = std::exp(tanh_argument);
            const auto exp_negative = std::exp(-tanh_argument);
            tanh_result = (exp_positive - exp_negative) / (exp_positive + exp_negative);
        }
        output[i] = 0.5f * x * (1.0f + tanh_result);
    }

    return output;
}

constexpr auto solution =
    // GeluReference;
    // GeluConstsFactoredOut;
    // GeluWithTanhViaExp;
    // GeluOMP_Basic;
    // GeluOMP_LoopUnrolling;
    GeluOMP; // added SIMD

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
    solution(input); // warming up
    float max_absolute_error = 0.f;
    float max_relative_error = 0.f;

    std::vector<double> time_list;
    for (int experiment_id = 0; experiment_id < NUM_EXPERIMENTS; ++experiment_id) {
        const auto start = std::chrono::high_resolution_clock::now();
        const auto result = solution(input);
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
