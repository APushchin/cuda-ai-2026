#include "gelu_omp.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdlib.h>
#include <stdio.h>

#undef GELU_USE_AVX2

#ifdef __x86_64__
#pragma GCC optimize("O3")
#pragma GCC target("avx2,fma")
#include "immintrin.h"
#define GELU_USE_AVX2 1
#endif

#ifdef GELU_USE_AVX2
__m256 _my256_exp_ps(__m256 x) {
    /* Modified code from this source: https://github.com/reyoung/avx_mathfun

    AVX implementation of exp
    Based on "sse_mathfun.h", by Julien Pommier
    http://gruntthepeon.free.fr/ssemath/
    Copyright (C) 2012 Giovanni Garberoglio
    Interdisciplinary Laboratory for Computational Science (LISC)
    Fondazione Bruno Kessler and University of Trento
    via Sommarive, 18
    I-38123 Trento (Italy)
    This software is provided 'as-is', without any express or implied
    warranty.  In no event will the authors be held liable for any damages
    arising from the use of this software.
    Permission is granted to anyone to use this software for any purpose,
    including commercial applications, and to alter it and redistribute it
    freely, subject to the following restrictions:
    1. The origin of this software must not be misrepresented; you must not
        claim that you wrote the original software. If you use this software
        in a product, an acknowledgment in the product documentation would be
        appreciated but is not required.
    2. Altered source versions must be plainly marked as such, and must not be
        misrepresented as being the original software.
    3. This notice may not be removed or altered from any source distribution.
    (this is the zlib license)

    */
    /*
    To increase the compatibility across different compilers the original code is
    converted to plain AVX2 intrinsics code without ingenious macro's,
    gcc style alignment attributes etc.
    Moreover, the part "express exp(x) as exp(g + n*log(2))" has been significantly simplified.
    This modified code is not thoroughly tested!
    */
    __m256   exp_hi        = _mm256_set1_ps(88.3762626647949f);
    __m256   exp_lo        = _mm256_set1_ps(-88.3762626647949f);

    __m256   cephes_LOG2EF = _mm256_set1_ps(1.44269504088896341f);
    __m256   inv_LOG2EF    = _mm256_set1_ps(0.693147180559945f);

    __m256   cephes_exp_p0 = _mm256_set1_ps(1.9875691500E-4);
    __m256   cephes_exp_p1 = _mm256_set1_ps(1.3981999507E-3);
    __m256   cephes_exp_p2 = _mm256_set1_ps(8.3334519073E-3);
    __m256   cephes_exp_p3 = _mm256_set1_ps(4.1665795894E-2);
    __m256   cephes_exp_p4 = _mm256_set1_ps(1.6666665459E-1);
    __m256   cephes_exp_p5 = _mm256_set1_ps(5.0000001201E-1);
    __m256   fx;
    __m256i  imm0;
    __m256   one           = _mm256_set1_ps(1.0f);

    x     = _mm256_min_ps(x, exp_hi);
    x     = _mm256_max_ps(x, exp_lo);

/* express exp(x) as exp(g + n*log(2)) */
    fx     = _mm256_mul_ps(x, cephes_LOG2EF);
    fx     = _mm256_round_ps(fx, _MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC);
    __m256  z = _mm256_mul_ps(fx, inv_LOG2EF);

    x      = _mm256_sub_ps(x, z);
    z      = _mm256_mul_ps(x, x);

    __m256 y = cephes_exp_p0;
    y      = _mm256_fmadd_ps(y, x, cephes_exp_p1);
    y      = _mm256_fmadd_ps(y, x, cephes_exp_p2);
    y      = _mm256_fmadd_ps(y, x, cephes_exp_p3);
    y      = _mm256_fmadd_ps(y, x, cephes_exp_p4);
    y      = _mm256_fmadd_ps(y, x, cephes_exp_p5);
    y      = _mm256_fmadd_ps(y, z, _mm256_add_ps(x, one));

/* build 2^n */
    imm0   = _mm256_cvttps_epi32(fx);
    imm0   = _mm256_add_epi32(imm0, _mm256_set1_epi32(0x7f));
    imm0   = _mm256_slli_epi32(imm0, 23);
    __m256  pow2n  = _mm256_castsi256_ps(imm0);
    y      = _mm256_mul_ps(y, pow2n);
    return y;
}
#endif

void GeluOMPMuchFaster(const std::vector<float>& input, std::vector<float>& output) {
    size_t n = input.size();
    output.resize(n);
    size_t nslices = 64;

    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < nslices; i++) {
        size_t j0 = i*n/nslices, j1 = (i+1)*n/nslices, j = j0;
        constexpr float argscale = std::sqrt(2.f/M_PI);
        const float* inptr = input.data();
        float* outptr = output.data();
    #ifdef GELU_USE_AVX2
        for (; j < j1 && (size_t)(void*)(outptr + j) % 32 != 0; j++) {
            float x = input[j];
            float y = argscale*x*(1.f + 0.044715f*x*x);
            y = 0.5f*x*(1 + (y >= 0.f ? 1.f : -1.f)*std::tanh(std::abs(y)));
            output[j] = y;
        }
        constexpr size_t VECSIZE = 8u;
        __m256 vscale = _mm256_set1_ps(-2.f*argscale);
        __m256 v0044715 = _mm256_set1_ps(0.044715f);
        __m256 one = _mm256_set1_ps(1.f), half = _mm256_set1_ps(0.5f);
        __m256 signbit = _mm256_castsi256_ps(_mm256_set1_epi32(0x80000000));
        for (; j + VECSIZE <= j1; j += VECSIZE) {
            __m256 x = _mm256_loadu_ps(inptr + j);
            __m256 y = _mm256_mul_ps(x, _mm256_fmadd_ps(_mm256_mul_ps(x, x), v0044715, one));
            __m256 ysign = _mm256_and_ps(y, signbit);
            __m256 absy = _mm256_andnot_ps(signbit, y);
            y = _my256_exp_ps(_mm256_mul_ps(vscale, absy));
            y = _mm256_div_ps(_mm256_sub_ps(one, y), _mm256_add_ps(one, y));
            y = _mm256_or_ps(y, ysign);
            y = _mm256_mul_ps(_mm256_mul_ps(x, half), _mm256_add_ps(one, y));
            _mm256_stream_ps(outptr + j, y);
        }
    #endif
        for (; j < j1; j++) {
            float x = input[j];
            float y = argscale*x*(1.f + 0.044715f*x*x);
            y = 0.5f*x*(1 + (y >= 0.f ? 1.f : -1.f)*std::tanh(std::abs(y)));
            output[j] = y;
        }
    }
}

std::vector<float> GeluOMP(const std::vector<float>& input) {
    std::vector<float> output;
    GeluOMPMuchFaster(input, output);
    return output;
}

#ifdef VP_RUN_TEST

std::vector<float> GeluOMPref(const std::vector<float>& input) {
    size_t n = input.size();
    std::vector<float> output(n);
    size_t nslices = 64;

    #pragma omp parallel for
    for (size_t i = 0; i < nslices; i++) {
        size_t j0 = i*n/nslices, j1 = (i+1)*n/nslices, j = j0;
        constexpr float argscale = std::sqrt(2.f/M_PI);
        const float* inptr = input.data();
        float* outptr = output.data();
        for (; j < j1; j++) {
            float x = inptr[j];
            float y = 0.5f*x*(1 + std::tanh(argscale*x*(1.f + 0.044715f*x*x)));
            outptr[j] = y;
        }
    }

    return output;
}

int main() {
    size_t n = 134217728u;
    std::vector<float> x(n);
    for (size_t i = 0; i < n; i++) {
        x[i] = ((float)rand()/RAND_MAX)*20.f - 10.f;
    }

    auto y = GeluOMP(x);
    int s = 0;
    for (size_t i = 0; i < n; i++) {
        s += y[i] > -5.f;
    }

    std::vector<float> yref = GeluOMPref(x);
    float err = 0.f;
    for (size_t i = 0; i < n; i++) {
        err = std::max(err, std::abs(y[i] - yref[i]));
    }
    printf("max absolute error = %.5g\n", err);

    // Performance Measuring
    std::vector<double> time_list;
    for (int i = 0; i < 10; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
    #if 0
        std::vector<float> y = GeluOMP(x);
    #else
        GeluOMPMuchFaster(x, y);
    #endif
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;
        time_list.push_back(duration.count());
        for (size_t i = 0; i < n; i++) {
            s += y[i] > -5.f;
        }
    }
    double time = *std::min_element(time_list.begin(), time_list.end());
    printf("time = %.2f\n", time);

    return s > 0;
}

#endif
