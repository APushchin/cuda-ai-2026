#include <cuda/cmath>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdlib.h>
#include <stdio.h>

#include "block_gemm_cuda.h"

constexpr int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;

__global__ void vecBlockGemm(const float* A, const float* B, float* C, int n) {
    int by = blockIdx.y, bx = blockIdx.x;
    int ty = threadIdx.x / (BN / TN);
    int tx = threadIdx.x % (BN / TN);

    __shared__ float As[BK][BM];
    __shared__ float Bs[BK][BN];

    A += by * BM * n;
    B += bx * BN;
    C += by * BM * n + bx * BN;

    int iAy = threadIdx.x / (BK/4);
    int iAx4 = (threadIdx.x % (BK/4)) * 4;
    int iBy = threadIdx.x / (BN/4);
    int iBx4 = (threadIdx.x % (BN/4)) * 4;

    float acc[TM * TN] = {0.f};
    float regM[TM];
    float regN[TN];

    for (int bk = 0; bk < n; bk += BK) {
        float4 a = *reinterpret_cast<const float4*>(&A[iAy * n + iAx4]);
        As[iAx4 + 0][iAy] = a.x;
        As[iAx4 + 1][iAy] = a.y;
        As[iAx4 + 2][iAy] = a.z;
        As[iAx4 + 3][iAy] = a.w;

        *reinterpret_cast<float4*>(&Bs[iBy][iBx4]) = *reinterpret_cast<const float4*>(&B[iBy * n + iBx4]);

        __syncthreads();
        A += BK;
        B += BK * n;

        #pragma unroll
        for (int k = 0; k < BK; k++) {
            #pragma unroll
            for (int i = 0; i < TM; i++) regM[i] = As[k][ty * TM + i];
            #pragma unroll
            for (int j = 0; j < TN; j++) regN[j] = Bs[k][tx * TN + j];
            #pragma unroll
            for (int i = 0; i < TM; i++)
                #pragma unroll
                for (int j = 0; j < TN; ++j)
                    acc[i * TN + j] += regM[i] * regN[j];
        }
        __syncthreads();
    }

    #pragma unroll
    for (int di = 0; di < TM; di++)
        #pragma unroll
        for (int dj = 0; dj < TN; dj++) {
            int i = ty * TM + di;
            int j = tx * TN + dj;
            C[i*n + j] = acc[di*TN + dj];
        }
}

std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    float* A = nullptr;
    float* B = nullptr;
    float* C = nullptr;

    int bytes = n * n * sizeof(float);
    cudaMalloc(&A, bytes);
    cudaMalloc(&B, bytes);
    cudaMalloc(&C, bytes);

    cudaMemcpy(A, a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(B, b.data(), bytes, cudaMemcpyHostToDevice);

    dim3 threads(256);
    dim3 blocks(n / BN, n / BM);
    vecBlockGemm<<<blocks, threads>>>(A, B, C, n);

    std::vector<float> c(n * n);
    cudaMemcpy(c.data(), C, bytes, cudaMemcpyDeviceToHost);

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);

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
    auto c = BlockGemmCUDA(a, b, n);
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
        auto c = BlockGemmCUDA(a, b, n);
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
