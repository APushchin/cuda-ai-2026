#include "gelu_omp.h"

#include <cmath>
#include <omp.h>

std::vector<float> GeluOMP(const std::vector<float>& input) {
    std::vector<float> result(input.size());
    if (input.empty()) return result;

    const float* in = input.data();
    float* out = result.data();
    const size_t n = input.size();

    const float sqrt_2_over_pi = std::sqrt(2.0f / static_cast<float>(M_PI));
    const float coeff = 0.044715f;
    const float neg_two = -2.0f;

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) {
        const float x = in[i];
        const float x3 = x * x * x;
        const float arg = neg_two * sqrt_2_over_pi * (x + coeff * x3);
        out[i] = x / (1.0f + std::exp(arg));
    }

    return result;
}