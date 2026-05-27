#include <cublas_v2.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdlib.h>
#include <stdio.h>

#include "gemm_cublas.h"

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                              const std::vector<float>& b,
                              int n) {
    float* A = nullptr;
    float* B = nullptr;
    float* C = nullptr;

    int bytes = n * n * sizeof(float);
    cudaMalloc(&A, bytes);
    cudaMalloc(&B, bytes);
    cudaMalloc(&C, bytes);

    cudaHostRegister((void*)a.data(), bytes, cudaHostRegisterDefault);
    cudaHostRegister((void*)b.data(), bytes, cudaHostRegisterDefault);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaMemcpyAsync(A, a.data(), bytes, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(B, b.data(), bytes, cudaMemcpyHostToDevice, stream);

    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetStream(handle, stream);

    const float alpha = 1.0f;
    const float beta  = 0.0f;

    cublasSgemm(
        handle, 
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        n, n, n,
        &alpha,
        B, n,
        A, n,
        &beta,
        C, n
    );

    std::vector<float> c(n * n);

    cudaMemcpyAsync(c.data(), C, bytes, cudaMemcpyDeviceToHost, stream);

    cudaStreamSynchronize(stream);
    
    cudaHostUnregister((void*)a.data());
    cudaHostUnregister((void*)b.data());

    cublasDestroy(handle);
    cudaStreamDestroy(stream);
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);

    return c;
}

#if 0
std::vector<float> NaiveGemmRef(const std::vector<float>& a,
                                const std::vector<float>& b,
                                int n) {
    std::vector<float> c(n * n, 0.f);

    const float* aptr = a.data();
    const float* bptr = b.data();
    float* cptr = c.data();

    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < n; k++) {
            float aval = aptr[i * n + k];
            for (int j = 0; j < n; j++) {
                cptr[i * n + j] += aval * bptr[k * n + j];
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
        a[i] = ((float)rand()/RAND_MAX)*20.f - 10.f;
        b[i] = ((float)rand()/RAND_MAX)*20.f - 10.f;
    }

    // Warming-up
    auto c = GemmCUBLAS(a, b, n);

    auto cref = NaiveGemmRef(a, b, n);
    float err = 0.f;
    for (size_t i = 0; i < n; i++) {
        err = std::max(err, std::abs(c[i] - cref[i]));
    }
    printf("max absolute error = %.5g\n", err);
    
    // Performance Measuring
    std::vector<double> time_list;
    for (int i = 0; i < 4; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        auto c = GemmCUBLAS(a, b, n);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;
        time_list.push_back(duration.count());
    }
    double time = *std::min_element(time_list.begin(), time_list.end());
    printf("time = %.4f\n", time);

    return 0;
}
#endif
