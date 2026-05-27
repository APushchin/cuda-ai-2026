#include "gelu_omp.h"

#include <omp.h>

#include <cmath>

std::vector<float> GeluRef(const std::vector<float> &input) {
    constexpr float sqrt_2_over_pi = 0.7978845608f;
    constexpr float coef = 0.044715f;

    const size_t size = input.size();
    std::vector<float> output(size);

    const float *__restrict in_ptr = input.data();
    float *__restrict out_ptr = output.data();

    #pragma omp parallel for simd
    for (size_t i = 0; i < size; ++i) {
        const float x = in_ptr[i];
        const float x_cube = x * x * x;
        const float inner = sqrt_2_over_pi * (x + coef * x_cube);
        out_ptr[i] = 0.5f * x * (1.0f + std::tanh(inner));
    }

    return output;
}

inline float tanh(float x) {
    return 1.f - (2.f / (1.f + std::exp(x * 2.f)));
}

std::vector<float> GeluOMP(const std::vector<float> &input) {
    constexpr float sqrt_2_over_pi = 0.7978845608f;
    constexpr float coef = 0.044715f;

    const size_t size = input.size();
    std::vector<float> output(size);

    const float *__restrict in_ptr = input.data();
    float *__restrict out_ptr = output.data();

    #pragma omp parallel for simd
    for (size_t i = 0; i < size; ++i) {
        const float x = in_ptr[i];
        const float x_cube = x * x * x;
        const float inner = sqrt_2_over_pi * (x + coef * x_cube);
        out_ptr[i] = 0.5f * x * (1.0f + tanh(inner));
    }

    return output;
}
