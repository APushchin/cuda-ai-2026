#include "gelu_omp.h"
#include <math.h>
#include <omp.h>
#include <immintrin.h>

// Took from : https://github.com/reyoung/avx_mathfun/blob/master/avx_mathfun.h
#define _PI32_CONST256(Name, Val)                                            \
  static const  int _pi32_256_##Name[8] __attribute__((aligned(32))) = { Val, Val, Val, Val, Val, Val, Val, Val }
#define _PS256_CONST(Name, Val)                                            \
    static const float _ps256_##Name[8] __attribute__((aligned(32))) = { Val, Val, Val, Val, Val, Val, Val, Val }
_PS256_CONST(1  , 1.0f);
_PS256_CONST(0p5, 0.5f);
_PS256_CONST(exp_hi,	88.3762626647949f);
_PS256_CONST(exp_lo,	-88.3762626647949f);

_PS256_CONST(cephes_LOG2EF, 1.44269504088896341);
_PS256_CONST(cephes_exp_C1, 0.693359375);
_PS256_CONST(cephes_exp_C2, -2.12194440e-4);

_PS256_CONST(cephes_exp_p0, 1.9875691500E-4);
_PS256_CONST(cephes_exp_p1, 1.3981999507E-3);
_PS256_CONST(cephes_exp_p2, 8.3334519073E-3);
_PS256_CONST(cephes_exp_p3, 4.1665795894E-2);
_PS256_CONST(cephes_exp_p4, 1.6666665459E-1);
_PS256_CONST(cephes_exp_p5, 5.0000001201E-1);

_PI32_CONST256(0x7f, 0x7f);

__m256 exp256_ps(__m256 x)
{
    __m256 tmp = _mm256_setzero_ps(), fx;
    __m256i imm0;
    __m256 one = *(__m256*)_ps256_1;

    x = _mm256_min_ps(x, *(__m256*)_ps256_exp_hi);
    x = _mm256_max_ps(x, *(__m256*)_ps256_exp_lo);

    /* express exp(x) as exp(g + n*log(2)) */
    fx = _mm256_mul_ps(x, *(__m256*)_ps256_cephes_LOG2EF);
    fx = _mm256_add_ps(fx, *(__m256*)_ps256_0p5);

    /* how to perform a floorf with SSE: just below */
    //imm0 = _mm256_cvttps_epi32(fx);
    //tmp  = _mm256_cvtepi32_ps(imm0);

    tmp = _mm256_floor_ps(fx);

    /* if greater, substract 1 */
    //__m256 mask = _mm256_cmpgt_ps(tmp, fx);    
    __m256 mask = _mm256_cmp_ps(tmp, fx, _CMP_GT_OS);    
    mask = _mm256_and_ps(mask, one);
    fx = _mm256_sub_ps(tmp, mask);

    tmp = _mm256_mul_ps(fx, *(__m256*)_ps256_cephes_exp_C1);
    __m256 z = _mm256_mul_ps(fx, *(__m256*)_ps256_cephes_exp_C2);
    x = _mm256_sub_ps(x, tmp);
    x = _mm256_sub_ps(x, z);

    z = _mm256_mul_ps(x,x);

    __m256 y = *(__m256*)_ps256_cephes_exp_p0;
    y = _mm256_mul_ps(y, x);
    y = _mm256_add_ps(y, *(__m256*)_ps256_cephes_exp_p1);
    y = _mm256_mul_ps(y, x);
    y = _mm256_add_ps(y, *(__m256*)_ps256_cephes_exp_p2);
    y = _mm256_mul_ps(y, x);
    y = _mm256_add_ps(y, *(__m256*)_ps256_cephes_exp_p3);
    y = _mm256_mul_ps(y, x);
    y = _mm256_add_ps(y, *(__m256*)_ps256_cephes_exp_p4);
    y = _mm256_mul_ps(y, x);
    y = _mm256_add_ps(y, *(__m256*)_ps256_cephes_exp_p5);
    y = _mm256_mul_ps(y, z);
    y = _mm256_add_ps(y, x);
    y = _mm256_add_ps(y, one);

    /* build 2^n */
    imm0 = _mm256_cvttps_epi32(fx);
    // another two AVX2 instructions
    imm0 = _mm256_add_epi32(imm0, *(__m256i*)_pi32_256_0x7f);
    imm0 = _mm256_slli_epi32(imm0, 23);
    __m256 pow2n = _mm256_castsi256_ps(imm0);
    y = _mm256_mul_ps(y, pow2n);
    return y;
}


std::vector<float> GeluOMP(const std::vector<float>& input)
{
    std::vector<float> res(input.size());
    const float* iptr = input.data();
    float* rptr = res.data();
    const float r1 = std::sqrt(2 / M_PI);
    int siz = (int)input.size();
#if 0 
    for(int elnum = 0; elnum < siz; elnum++)
    {
        float x = iptr[elnum];
        float targ = r1*(x+0.044715*x*x*x);
        float th = 1 - 2. / (std::exp(2*targ)+1);
        rptr[elnum] = 0.5*x*(1.+th);
    }
#elif 0
    #pragma omp parallel for
    for(int elnum = 0; elnum < siz; elnum++)
    {
        float x = iptr[elnum];
        float targ = r1*(x+0.044715*x*x*x);
        float th = 1 - 2. / (std::exp(2*targ)+1);
        rptr[elnum] = 0.5*x*(1.+th);
    }
#else
    const int threadnums = 4;
    const int lanes = 8;
    int tasks = std::max(1, siz/threadnums);
    int stride = siz/threadnums;
    stride -= stride%lanes;
    stride = std::max(lanes, stride);
    #pragma omp parallel for
    for(int tnum = 0; tnum  < threadnums; tnum++)
    {
        int start = tnum*stride;
        int end = (tnum+1)*stride;
        end = std::min(end, siz);
        int elnum = start;
        __m256 v_r1 = _mm256_set1_ps(2*r1);
        __m256 v_c1 = _mm256_set1_ps(0.044715f);
        __m256 v_one = _mm256_set1_ps(1.0f);
        __m256 v_two = _mm256_set1_ps(2.0f);
        __m256 v_half = _mm256_set1_ps(0.5f);
        for(; elnum <= end - lanes; elnum += lanes)
        {
            // float x = iptr[elnum];
            __m256 x = _mm256_loadu_ps(iptr + elnum);
            // float targ = r1*(x+0.044715*x*x*x);
            __m256 x3 = _mm256_mul_ps(_mm256_mul_ps(x, x), x);
            __m256 targ = _mm256_mul_ps(v_r1, _mm256_fmadd_ps(v_c1, x3, x));//DUBUG: looks like x, v_c1, x3 actually...
            // th = 1 - 2 / (exp(2*targ) + 1)
            __m256 e_val = exp256_ps(targ);
            __m256 th = _mm256_sub_ps(v_one, _mm256_div_ps(v_two, _mm256_add_ps(e_val, v_one)));
            // res = 0.5 * x * (1 + th)
            __m256 res = _mm256_mul_ps(v_half, _mm256_mul_ps(x, _mm256_add_ps(v_one, th)));
            _mm256_storeu_ps(rptr + elnum, res);
        }
        for(; elnum < end; elnum++)
        {
            float x = iptr[elnum];
            float targ = r1*(x+0.044715*x*x*x);
            float th = 1 - 2. / (std::exp(2*targ)+1);
            rptr[elnum] = 0.5*x*(1.+th);
        }
    }

#endif
    return res;
}