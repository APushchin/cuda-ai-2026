#include <cuda/cmath>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdlib.h>
#include <stdio.h>
#include <cublas_v2.h>

#include "gemm_cublas.h"

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                              const std::vector<float>& b,
                              int n) {
    cublasHandle_t cublas; cublasCreate(&cublas);

    float* A = nullptr;
    float* B = nullptr;
    float* C = nullptr;

    int bytes = n * n * sizeof(float);
    cudaMalloc(&A, bytes);
    cudaMalloc(&B, bytes);
    cudaMalloc(&C, bytes);

    cudaMemcpy(A, a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(B, b.data(), bytes, cudaMemcpyHostToDevice);

    float one = 1.f, zero = 0.f;
    cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &one, B, n, A, n, &zero, C, n);

    std::vector<float> c(n * n);
    cudaMemcpy(c.data(), C, bytes, cudaMemcpyDeviceToHost);

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);

    cublasDestroy(cublas);

    return c;
}

#ifdef VP_RUN_TEST
std::vector<float> LessThanNaiveGemm(const std::vector<float>& a,
                                    const std::vector<float>& b,
                                    int n_) {
    constexpr int ntiles = 64;
    std::vector<float> c(n_*n_);

    #pragma omp parallel for
    for (int t = 0; t < ntiles; t++) {
        int n = n_;
        int i0 = t*n/ntiles, i1 = (t+1)*n/ntiles;
        for (int i = i0; i < i1; i++) {
            const float* aptr = &a[i*n];
            const float* bptr = b.data();
            float* cptr = &c[i*n];
            memset(cptr, 0, n*sizeof(cptr[0]));

            for (int k = 0; k < n; k++, bptr += n) {
                float aval = a[k];
                for (int j = 0; j < n; j++)
                    cptr[j] += aval*bptr[j];
            }
        }
    }

    return c;
}

int main() {
    size_t n = 4096;
    std::vector<float> a(n*n);
    std::vector<float> b(n*n);
    for (size_t i = 0; i < n*n; i++) {
        a[i] = ((float)rand()/RAND_MAX) - 0.5f;
        b[i] = ((float)rand()/RAND_MAX) - 0.5f;
    }

    // Warming-up
    auto c = GemmCUBLAS(a, b, n);
    auto cref = LessThanNaiveGemm(a, b, n);
    float err = 0.f;
    for (size_t i = 0; i < n; i++) {
        err = std::max(err, std::abs(c[i] - cref[i]));
    }
    printf("max absolute error = %.5g\n", err);

    // Performance Measuring
    std::vector<double> time_list;
    for (int i = 0; i < 4; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
    #if 1
        auto c = GemmCUBLAS(a, b, n);
    #else
        auto c = LessThanNaiveGemm(a, b, n);
    #endif
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;
        time_list.push_back(duration.count());
    }
    double time = *std::min_element(time_list.begin(), time_list.end());
    printf("time = %.4f\n", time);

    return 0;
}
#endif
