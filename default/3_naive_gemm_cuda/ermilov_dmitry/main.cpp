#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#include "naive_gemm_cuda.h"

std::vector<float> NaiveGemmRef(const std::vector<float>& a,
                                const std::vector<float>& b,
                                int n) {
    std::vector<float> c(n * n, 0.0f);

    for (int i = 0; i < n; ++i) {
        for (int k = 0; k < n; ++k) {
            float aik = a[i * n + k];
            for (int j = 0; j < n; ++j) {
                c[i * n + j] += aik * b[k * n + j];
            }
        }
    }

    return c;
}

template <typename Func>
double Benchmark(Func&& f, int iterations = 5) {
    double best = 1e100;

    for (int i = 0; i < iterations; ++i) {
        auto start = std::chrono::high_resolution_clock::now();

        auto result = f();

        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double> elapsed = end - start;
        best = std::min(best, elapsed.count());

        volatile float sink = result[result.size() / 2];
        (void)sink;
    }

    return best;
}

int main() {
    constexpr int N = 1024;

    std::vector<float> A(N * N);
    std::vector<float> B(N * N);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-5.f, 5.f);

    for (auto& x : A) {
        x = dist(rng);
    }

    for (auto& x : B) {
        x = dist(rng);
    }

    std::cout << "Matrix size: " << N << " x " << N << "\n\n";

    auto ref_time = Benchmark([&]() {
        return NaiveGemmRef(A, B, N);
    });

    auto cuda_time = Benchmark([&]() {
        return NaiveGemmCUDA(A, B, N);
    });

    auto ref = NaiveGemmRef(A, B, N);
    auto gpu = NaiveGemmCUDA(A, B, N);

    float max_error = 0.0f;

    for (size_t i = 0; i < ref.size(); ++i) {
        max_error = std::max(max_error, std::abs(ref[i] - gpu[i]));
    }

    std::cout << "Reference : " << ref_time << " s\n";
    std::cout << "CUDA      : " << cuda_time << " s\n";
    std::cout << "Speedup   : " << ref_time / cuda_time << "x\n";
    std::cout << "Max error : " << max_error << '\n';

    return 0;
}