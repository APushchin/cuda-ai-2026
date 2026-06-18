#include "gelu_omp.h"
#include <cmath>

inline float tahn_exp(float x) {
    float ex = std::exp(2.0f * x);
    return (ex - 1.0f) / (ex + 1.0f);
}

std::vector<float> GeluOMP(const std::vector<float>& input) {
    
    int size = input.size();
    std::vector<float> output(size);

    #pragma omp parallel for
    for (int i = 0; i < size; i++) {
        float x = input[i];
        output[i] = x * x * x;
        output[i] *= 0.044715f;
        output[i] += x;
        output[i] *= 0.79788456f;
        output[i] = tahn_exp(output[i]);
        output[i] += 1.0f;
        output[i] *= x;
        output[i] *= 0.5f;
    }

    return output;
}