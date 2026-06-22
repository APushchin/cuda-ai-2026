#include "gelu_omp.h"

#include <cmath>
#include <omp.h>

using namespace std;

inline float fastTanh(const float val) {
    if (abs(val) < 40.f) {
        const float tmp = std::exp(2.f * val);
        return (tmp - 1.f) / (tmp + 1.f);
    }
    return std::tanh(val);
}

vector<float> GeluOMP(const vector<float>& input) 
{
    const float const1 = std::sqrt(2.f / M_PI); 
    const float const2 = 0.044715f;
    const size_t size = input.size();

    vector<float> result(size);

    #pragma omp parallel for
    for (size_t idx = 0; idx < size; ++idx) {
        const float val = input[idx];
        float tanhResult = fastTanh(const1 * val * (1.f + const2 * val * val));
        result[idx] = val * 0.5f * (1.f + tanhResult);
    }

    return result;
}