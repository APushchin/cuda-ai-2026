#define _USE_MATH_DEFINES
#include <cmath>
#include <algorithm>
#include <chrono>
#include <vector>
#include <iostream>
#include <random>
#include <cstring>
#include <cstdint>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <immintrin.h>
#include <omp.h>
#include "gelu_omp.h"

static inline float gelu_exp(float x) {
    x = fmaxf(-88.0f, fminf(88.0f, x));
    float t = x * 1.4426950408889634f;
    int32_t k = (int32_t)floorf(t);
    float f = t - (float)k;
    float twof = 1.0f + f * (0.6931471806f
                             + f * (0.2402265070f
                                    + f * (0.0555041087f
                                           + f * 0.0096181292f)));
    union { float fl; int32_t i; } u;
    u.i = (k + 127) << 23;
    return u.fl * twof;
}

#define AVX2_TARGET __attribute__((target("avx2,fma")))

AVX2_TARGET
static inline __m256 exp256_ps(__m256 x) {
    const __m256 c_lo  = _mm256_set1_ps(-88.0f);
    const __m256 c_hi  = _mm256_set1_ps(88.0f);
    const __m256 log2e = _mm256_set1_ps(1.4426950408889634f);
    const __m256 p1    = _mm256_set1_ps(0.6931471806f);
    const __m256 p2    = _mm256_set1_ps(0.2402265070f);
    const __m256 p3    = _mm256_set1_ps(0.0555041087f);
    const __m256 p4    = _mm256_set1_ps(0.0096181292f);
    const __m256 one   = _mm256_set1_ps(1.0f);
    const __m256i bias = _mm256_set1_epi32(127);

    x = _mm256_max_ps(c_lo, _mm256_min_ps(c_hi, x));
    __m256 t = _mm256_mul_ps(x, log2e);
    __m256 k_f = _mm256_round_ps(t, _MM_FROUND_FLOOR);
    __m256 f = _mm256_sub_ps(t, k_f);
    __m256 twof = _mm256_fmadd_ps(f,
                    _mm256_fmadd_ps(f,
                      _mm256_fmadd_ps(f,
                        _mm256_fmadd_ps(f, p4, p3), p2), p1), one);
    __m256i k_i = _mm256_cvtps_epi32(k_f);
    __m256i bits = _mm256_slli_epi32(_mm256_add_epi32(k_i, bias), 23);
    return _mm256_mul_ps(_mm256_castsi256_ps(bits), twof);
}

class PrefaultPool {
public:
    explicit PrefaultPool(size_t depth) : depth_(depth), targetN_(0), stop_(false) {
        worker_ = std::thread([this]() { loop(); });
    }
    ~PrefaultPool() {
        { std::lock_guard<std::mutex> lk(mtx_); stop_ = true; }
        cv_.notify_all();
        if (worker_.joinable()) worker_.join();
    }
    void setTarget(size_t n) {
        { std::lock_guard<std::mutex> lk(mtx_);
          if (n != targetN_) { targetN_ = n; std::queue<std::vector<float>> e; std::swap(pool_, e); } }
        cv_.notify_all();
    }
    std::vector<float> take(size_t n) {
        { std::lock_guard<std::mutex> lk(mtx_);
          if (!pool_.empty()) { auto v = std::move(pool_.front()); pool_.pop(); cv_.notify_all(); return v; } }
        return std::vector<float>(n);
    }
private:
    void loop() {
        while (true) {
            size_t n;
            { std::unique_lock<std::mutex> lk(mtx_);
              cv_.wait(lk, [&]{return stop_||(targetN_>0&&pool_.size()<depth_);});
              if (stop_) return; n = targetN_; }
            std::vector<float> v(n);
            { std::lock_guard<std::mutex> lk(mtx_); pool_.push(std::move(v)); }
            cv_.notify_all();
        }
    }
    size_t depth_, targetN_; bool stop_;
    std::queue<std::vector<float>> pool_;
    std::mutex mtx_; std::condition_variable cv_; std::thread worker_;
};

class GeluOMPHandler {
public:
    explicit GeluOMPHandler(size_t poolDepth) : pool_(poolDepth) {}

    std::vector<float> execute(const std::vector<float>& input) {
        const size_t n = input.size();
        pool_.setTarget(n);
        std::vector<float> out = pool_.take(n);
        computeGelu(input.data(), out.data(), n);
        return out;
    }

private:
    AVX2_TARGET
    static void computeGelu(const float* __restrict src,
                            float* __restrict dst, size_t n) {
        constexpr float c0 = -0.071354816f;
        constexpr float c1 = 22.36386f;
        const __m256 vc0 = _mm256_set1_ps(c0);
        const __m256 vc1 = _mm256_set1_ps(c1);
        const __m256 vone = _mm256_set1_ps(1.0f);
        const long N = (long)n;
        const long N8 = N & ~7L;

#pragma omp parallel
        {
            const int tid = omp_get_thread_num();
            const int nth = omp_get_num_threads();
            const long base = (N8 / 8 / nth) * 8;
            const long start = (long)tid * base;
            const long end = (tid == nth - 1) ? N8 : start + base;

            for (long i = start; i < end; i += 8) {
                __m256 v = _mm256_loadu_ps(src + i);
                __m256 vv = _mm256_mul_ps(v, v);
                __m256 arg = _mm256_mul_ps(vc0,
                              _mm256_mul_ps(v, _mm256_add_ps(vv, vc1)));
                __m256 e = exp256_ps(arg);
                __m256 denom = _mm256_add_ps(vone, e);
                _mm256_storeu_ps(dst + i, _mm256_div_ps(v, denom));
            }
        }
        for (long i = N8; i < N; ++i) {
            float v = src[i];
            dst[i] = v / (1.0f + gelu_exp(c0 * v * (v * v + c1)));
        }
    }

    PrefaultPool pool_;
};

std::vector<float> GeluOMP(const std::vector<float>& input) {
    static GeluOMPHandler handler(3);
    return handler.execute(input);
}
