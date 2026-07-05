#include "gelu_omp.h"

#include <cmath>
#include <cstring>
#include <vector>

static inline float fast_exp_neg(float x) {
    constexpr float ln_min = std::log(std::numeric_limits<float>::min());
    constexpr float ln_max = std::log(std::numeric_limits<float>::max());
    constexpr float log2e   =  1.4426950408889634f;
    constexpr float ln2     =  0.6931471805599453f;

    if (x < ln_min) return 0.0f;
    if (x > ln_max) x = ln_max;

    float nf = x * log2e + 0.5f;
    int32_t n = static_cast<int32_t>(nf);
    float r = x - static_cast<float>(n) * ln2;

    int32_t exp_bits = (n + 127) << 23;
    float two_pow_n;
    std::memcpy(&two_pow_n, &exp_bits, sizeof(two_pow_n));

    const float c0 = 1.0f;
    const float c1 = 1.0f;
    const float c2 = 0.5f;
    const float c3 = 1.0f / 6.0f;
    const float c4 = 1.0f / 24.0f;
    const float c5 = 1.0f / 120.0f;
    const float c6 = 1.0f / 720.0f;
    const float c7 = 1.0f / 5040.0f;

    float exp_r = ((((((c7 * r + c6) * r + c5) * r + c4) * r + c3) * r + c2) * r + c1) * r + c0;

    return exp_r * two_pow_n;
}

std::vector<float> GeluOMP(const std::vector<float>& input) {
    const size_t n = input.size();
    std::vector<float> output(n);

    const float sqrt_2_pi  = 0.7978845608028654f;
    const float alpha      = 0.044715f;
    const float neg_two    = -2.0f;

    const float* __restrict p_in  = input.data();
    float*       __restrict p_out = output.data();

    #pragma omp parallel for simd
    for (int64_t i = 0; i < static_cast<int64_t>(n); i += 4) {
        const float in0 = p_in[i];
        const float in1 = (i + 1 < n) ? p_in[i + 1] : 0.0f;
        const float in2 = (i + 2 < n) ? p_in[i + 2] : 0.0f;
        const float in3 = (i + 3 < n) ? p_in[i + 3] : 0.0f;

        float z0 = sqrt_2_pi * (in0 + alpha * in0 * in0 * in0);
        float z1 = sqrt_2_pi * (in1 + alpha * in1 * in1 * in1);
        float z2 = sqrt_2_pi * (in2 + alpha * in2 * in2 * in2);
        float z3 = sqrt_2_pi * (in3 + alpha * in3 * in3 * in3);

        float e0 = fast_exp_neg(neg_two * z0);
        float e1 = fast_exp_neg(neg_two * z1);
        float e2 = fast_exp_neg(neg_two * z2);
        float e3 = fast_exp_neg(neg_two * z3);

        if (i < n)     p_out[i]     = in0 / (1.0f + e0);
        if (i + 1 < n) p_out[i + 1] = in1 / (1.0f + e1);
        if (i + 2 < n) p_out[i + 2] = in2 / (1.0f + e2);
        if (i + 3 < n) p_out[i + 3] = in3 / (1.0f + e3);
    }

    return output;
}