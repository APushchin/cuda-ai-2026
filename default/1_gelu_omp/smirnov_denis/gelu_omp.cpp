#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdlib.h>
#include <stdio.h>
#include "gelu_omp.h"

inline float fast_tanh(float x) {
    float e2x = std::exp(2.f * x);
    return (e2x - 1.f) / (e2x + 1.f);
}

std::vector<float> GeluOMP(const std::vector<float>& input) {
    size_t n = input.size();
    std::vector<float> output(n);

    const float* inptr = input.data();
    float* outptr = output.data();

    #pragma omp parallel for
    for (size_t i = 0; i < n; i++) {
        float x = inptr[i];
        float inner = 0.79788456f * x * (1.f + 0.044715f * x * x);
        outptr[i] = 0.5f * x * (1.f + fast_tanh(inner));
    }

    return output;
}

#if 0
std::vector<float> GeluOMPref(const std::vector<float>& input) {
    size_t n = input.size();
    std::vector<float> output(n);

    constexpr float argscale = std::sqrt(2.f/M_PI);
    const float* inptr = input.data();
    float* outptr = output.data();

    for (size_t i = 0; i < n; i++) {
        float x = input[i];
        float y = 0.5f*x*(1 + std::tanh(argscale*x*(1.f + 0.044715f*x*x)));
        output[i] = y;
    }

    return output;
}

int main() {
    size_t n = 134217728u;
    std::vector<float> x(n);
    for (size_t i = 0; i < n; i++) {
        x[i] = ((float)rand()/RAND_MAX)*20.f - 10.f;
    }

    // Warming-up
    auto y = GeluOMP(x);

    std::vector<float> yref = GeluOMPref(x);
    float err = 0.f;
    for (size_t i = 0; i < n; i++) {
        err = std::max(err, std::abs(y[i] - yref[i]));
    }
    printf("max absolute error = %.5g\n", err);

    // Performance Measuring
    std::vector<double> time_list;
    for (int i = 0; i < 4; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        auto y = GeluOMP(x);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;
        time_list.push_back(duration.count());
    }
    double time = *std::min_element(time_list.begin(), time_list.end());
    printf("time = %.2f\n", time);

    return 0;
}
#endif
