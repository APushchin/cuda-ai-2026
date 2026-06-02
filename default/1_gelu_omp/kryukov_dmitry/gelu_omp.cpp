#include "gelu_omp.h"
#include <vector>
#include <cmath>
#include <stdint.h>

#define K 0.044715f
#define COEFF std::sqrt(2.0f / 3.14159265358979323846f)

#ifdef FASTAPPROX
    /// from https://github.com/romeric/fastapprox/blob/master/fastapprox/src/fastexp.h
    // BEGIN borrowed from fastapprox
    // Note: This approximation degrades the MAE from 1e-9 to 1e-6.
    static inline float
    fastpow2 (float p)
    {
    float offset = (p < 0) ? 1.0f : 0.0f;
    float clipp = (p < -126) ? -126.0f : p;
    int w = clipp;
    float z = clipp - w + offset;
    union { uint32_t i; float f; } v = { static_cast<uint32_t> ( (1 << 23) * (clipp + 121.2740575f + 27.7280233f / (4.84252568f - z) - 1.49012907f * z) ) };

    return v.f;
    }

    static inline float
    fastexp (float p)
    {
    return fastpow2 (1.442695040f * p);
    }
    // END borrowed from fastapprox

    inline float exp_tanh(float x) {
        float exp_x = fastexp(2*x);
        return (exp_x - 1.0) / (exp_x + 1.0);
    }

#else
    inline float exp_tanh(float x) {
        float exp_x = std::exp(2*x);
        return (exp_x - 1.0) / (exp_x + 1.0);
    }
#endif

std::vector<float> GeluSEQ(const std::vector<float>& input) {
    std::vector<float> result(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        result[i] = 0.5f * input[i] * (1.0f + std::tanh(COEFF * (input[i] + K * input[i] * input[i] * input[i])));
    }
    return result;
}

std::vector<float> GeluOMP(const std::vector<float>& input) {
    const size_t n = input.size();
    std::vector<float> result;
    result.reserve(n);
    float *__restrict p_out = result.data();
    const float *__restrict p_in = input.data();
    #pragma omp parallel for simd schedule(static, 4096) aligned(p_in, p_out : 32)
    for (size_t i = 0; i < n; ++i) {
        p_out[i] = 0.5f * p_in[i] * (1.0f + exp_tanh(COEFF * (p_in[i]  + K * p_in[i] * p_in[i] * p_in[i])));
    }
    return result;
}
