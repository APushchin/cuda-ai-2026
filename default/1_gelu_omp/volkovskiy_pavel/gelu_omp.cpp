#include "gelu_omp.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <immintrin.h>

namespace {
static constexpr float C1 = 2.0f * std::sqrt(2.0f / M_PI);
static constexpr float C2 = C1 * 0.044715f;

float geluf_opt(float x)
{
    const float exp2x = std::exp((C1 + C2 * x * x) * x);
    return x * ((exp2x) / (exp2x + 1));
}

void geluf_array(const float* in, float* out, std::size_t count) {
    for (std::size_t i = 0; i < count; ++i) {
        *out++ = geluf_opt(*in++);
    }
}

__m256 exp256_ps(__m256 x) {
/* Modified code. The original code is here: https://github.com/reyoung/avx_mathfun

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
  gcc style alignment attributes etc. The modified code requires AVX2
*/
__m256   exp_hi        = _mm256_set1_ps(88.3762626647949f);
__m256   exp_lo        = _mm256_set1_ps(-88.3762626647949f);

__m256   cephes_LOG2EF = _mm256_set1_ps(1.44269504088896341);
__m256   cephes_exp_C1 = _mm256_set1_ps(0.693359375);
__m256   cephes_exp_C2 = _mm256_set1_ps(-2.12194440e-4);

__m256   cephes_exp_p0 = _mm256_set1_ps(1.9875691500E-4);
__m256   cephes_exp_p1 = _mm256_set1_ps(1.3981999507E-3);
__m256   cephes_exp_p2 = _mm256_set1_ps(8.3334519073E-3);
__m256   cephes_exp_p3 = _mm256_set1_ps(4.1665795894E-2);
__m256   cephes_exp_p4 = _mm256_set1_ps(1.6666665459E-1);
__m256   cephes_exp_p5 = _mm256_set1_ps(5.0000001201E-1);
__m256   tmp           = _mm256_setzero_ps(), fx;
__m256i  imm0;
__m256   one           = _mm256_set1_ps(1.0f);

        x     = _mm256_min_ps(x, exp_hi);
        x     = _mm256_max_ps(x, exp_lo);

  /* express exp(x) as exp(g + n*log(2)) */
        fx    = _mm256_mul_ps(x, cephes_LOG2EF);
        fx    = _mm256_add_ps(fx, _mm256_set1_ps(0.5f));
        tmp   = _mm256_floor_ps(fx);
__m256  mask  = _mm256_cmp_ps(tmp, fx, _CMP_GT_OS);    
        mask  = _mm256_and_ps(mask, one);
        fx    = _mm256_sub_ps(tmp, mask);
        tmp   = _mm256_mul_ps(fx, cephes_exp_C1);
__m256  z     = _mm256_mul_ps(fx, cephes_exp_C2);
        x     = _mm256_sub_ps(x, tmp);
        x     = _mm256_sub_ps(x, z);
        z     = _mm256_mul_ps(x,x);

__m256  y     = cephes_exp_p0;
        y     = _mm256_mul_ps(y, x);
        y     = _mm256_add_ps(y, cephes_exp_p1);
        y     = _mm256_mul_ps(y, x);
        y     = _mm256_add_ps(y, cephes_exp_p2);
        y     = _mm256_mul_ps(y, x);
        y     = _mm256_add_ps(y, cephes_exp_p3);
        y     = _mm256_mul_ps(y, x);
        y     = _mm256_add_ps(y, cephes_exp_p4);
        y     = _mm256_mul_ps(y, x);
        y     = _mm256_add_ps(y, cephes_exp_p5);
        y     = _mm256_mul_ps(y, z);
        y     = _mm256_add_ps(y, x);
        y     = _mm256_add_ps(y, one);

  /* build 2^n */
        imm0  = _mm256_cvttps_epi32(fx);
        imm0  = _mm256_add_epi32(imm0, _mm256_set1_epi32(0x7f));
        imm0  = _mm256_slli_epi32(imm0, 23);
__m256  pow2n = _mm256_castsi256_ps(imm0);
        y     = _mm256_mul_ps(y, pow2n);
        return y;
}
} // namespace

std::vector<float> GeluOMP(const std::vector<float>& input) {
    const std::size_t length = input.size();
    std::vector<float> result(length, 0);

    uintptr_t input_misalign = (32 - ((const uintptr_t)input.data() & 31)) % 32;
    size_t head = input_misalign / sizeof(float);

    size_t result_misalignment = 32 - ((const uintptr_t)(result.data() + head) & 31);
    bool is_result_aligned = (result_misalignment == 0 || result_misalignment == 32);

    if (head >= length) {
        geluf_array(&input[0], &result[0], length);
    } else {
        geluf_array(&input[0], &result[0], head);

        const std::size_t tail_size = (length - head) % 8;
        const std::size_t tail = length - tail_size;

        __m256 vC1 = _mm256_set1_ps(C1);
        __m256 vC2 = _mm256_set1_ps(C2);
        __m256 vone = _mm256_set1_ps(1.0f);

       if (is_result_aligned) {
            #pragma omp parallel for simd
            for (std::size_t i = head; i < tail; i += 8) {
                __m256 vx = _mm256_load_ps(&input[i]);
                // exp(C1 + C2 * x * x) * x;
                __m256 vx2 = _mm256_mul_ps(vx, vx);
                __m256 vx2ss = _mm256_fmadd_ps(vC2, vx2, vC1);
                __m256 exparg = _mm256_mul_ps(vx2ss, vx);
                __m256 vexp2x = exp256_ps(exparg);
                // x * (exp2x / (exp2x + 1));
                __m256 vexp2xplus1 = _mm256_add_ps(vexp2x, vone);
                __m256 vdivres = _mm256_div_ps(vexp2x, vexp2xplus1);
                __m256 vres = _mm256_mul_ps(vdivres, vx);
                _mm256_stream_ps(&result[i], vres);
            }
        } else {
            #pragma omp parallel for simd
            for (std::size_t i = head; i < tail; i += 8) {
                __m256 vx = _mm256_load_ps(&input[i]);
                // exp(C1 + C2 * x * x) * x;
                __m256 vx2 = _mm256_mul_ps(vx, vx);
                __m256 vx2ss = _mm256_fmadd_ps(vC2, vx2, vC1);
                __m256 exparg = _mm256_mul_ps(vx2ss, vx);
                __m256 vexp2x = exp256_ps(exparg);
                // x * (exp2x / (exp2x + 1));
                __m256 vexp2xplus1 = _mm256_add_ps(vexp2x, vone);
                __m256 vdivres = _mm256_div_ps(vexp2x, vexp2xplus1);
                __m256 vres = _mm256_mul_ps(vdivres, vx);
                _mm256_storeu_ps(&result[i], vres);
            }
        }

        geluf_array(&input[tail], &result[tail], tail_size);
    }

    return result;
}

