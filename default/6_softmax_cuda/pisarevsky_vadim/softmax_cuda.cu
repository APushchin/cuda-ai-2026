#include <cuda/cmath>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <float.h>
#include <limits>
#include <stdlib.h>
#include <stdio.h>

#include "softmax_cuda.h"

#define BLOCK_SIZE 256
#define WARP_SIZE 32
#define NUM_WARPS (BLOCK_SIZE / WARP_SIZE)

__device__ __forceinline__
void warp_reduce(float& m, float& s) {
    #pragma unroll
    for (int k = WARP_SIZE/2; k > 0; k >>= 1) {
        float m2 = __shfl_down_sync(0xffffffff, m, k);
        float s2 = __shfl_down_sync(0xffffffff, s, k);
        float new_max = fmaxf(m, m2);
        s = s * __expf(m - new_max) + s2 * __expf(m2 - new_max);
        m = new_max;
    }
}

__global__ void vecSoftmax(const float* X, float* Y, int nrows, int ncols) {
    int row = blockIdx.x;
    if (row >= nrows) return;

    int tid = threadIdx.x;
    const float4* X4 = (const float4*)(X + row*ncols);
    float4* Y4 = (float4*)(Y + row*ncols);
    int ncols_4 = ncols / 4;

    float t_max = -FLT_MAX, t_sum = 0.f;
    for (int j = tid; j < ncols_4; j += BLOCK_SIZE) {
        float4 v = X4[j];

        float vmax = fmaxf(fmaxf(v.x, v.y), fmaxf(v.z, v.w));
        float vsum = __expf(v.x - vmax) + __expf(v.y - vmax) +
                     __expf(v.z - vmax) + __expf(v.w - vmax);
        float new_max = fmaxf(t_max, vmax);
        t_sum = t_sum * __expf(t_max - new_max) + vsum * __expf(vmax - new_max);
        t_max = new_max;
    }

    __shared__ float wmax[WARP_SIZE], wsum[WARP_SIZE];
    int lane = tid & (WARP_SIZE - 1), wid = tid / WARP_SIZE;

    warp_reduce(t_max, t_sum);
    if (lane == 0) {
        wmax[wid] = t_max;
        wsum[wid] = t_sum;
    }
    __syncthreads();

    if (wid == 0) {
        t_max = lane < NUM_WARPS ? wmax[lane] : -FLT_MAX;
        t_sum = lane < NUM_WARPS ? wsum[lane] : 0.0f;
        warp_reduce(t_max, t_sum);
        if (lane == 0) { wmax[0] = t_max; wsum[0] = t_sum; }
    }
    __syncthreads();
    float maxval = wmax[0];
    float scale = 1.f/wsum[0];

    for (int j = tid; j < ncols_4; j += BLOCK_SIZE) {
        float4 v = X4[j];
        float4 r = { __expf(v.x - maxval) * scale, __expf(v.y - maxval) * scale,
                     __expf(v.z - maxval) * scale, __expf(v.w - maxval) * scale };
        Y4[j] = r;
    }
}

std::vector<float> SoftmaxCUDA(const std::vector<float>& input, int nrows) {
    int ncols = int(input.size() / (size_t)nrows);
    size_t nbytes = input.size() * sizeof(input[0]);

    float* X = nullptr;
    cudaMalloc(&X, nbytes);
    cudaMemcpy(X, input.data(), nbytes, cudaMemcpyHostToDevice);

    vecSoftmax<<<nrows, BLOCK_SIZE>>>(X, X, nrows, ncols);

    std::vector<float> output(input.size());
    cudaMemcpy(output.data(), X, nbytes, cudaMemcpyDeviceToHost);
    cudaFree(X);

    return output;
}

#ifdef VP_RUN_TEST
std::vector<float> SoftmaxRef(const std::vector<float>& input, int nrows_) {
    constexpr int ntiles = 64;
    std::vector<float> output(input.size());

    #pragma omp parallel for
    for (int t = 0; t < ntiles; t++) {
        int nrows = nrows_;
        int ncols = int(input.size() / (size_t)nrows);
        int i0 = t*nrows/ntiles, i1 = (t+1)*nrows/ntiles;
        for (int i = i0; i < i1; i++) {
            const float* inptr = &input[i*ncols];
            float* outptr = &output[i*ncols];
            float maxval = inptr[0];
            for (int j = 1; j < ncols; j++)
                maxval = std::max(maxval, outptr[j]);
            double sum = 0.;
            for (int j = 0; j < ncols; j++) {
                float v = std::exp(inptr[j] - maxval);
                outptr[j] = v;
                sum += v;
            }
            float scale = float(1./sum);
            for (int j = 0; j < ncols; j++)
                outptr[j] *= scale;
        }
    }

    return output;
}

int main() {
    size_t nrows = 8192;
    size_t ncols = 16384;
    size_t data_size = nrows * ncols;
    std::vector<float> input(data_size);
    for (size_t i = 0; i < data_size; i++) {
        input[i] = ((float)rand()/RAND_MAX)*20.f - 10.f;
    }

    // Warming-up
    auto output = SoftmaxCUDA(input, nrows);

    std::vector<float> outref = SoftmaxRef(input, nrows);
    float err = 0.f;
    size_t nbad = 0;
    for (size_t i = 0; i < data_size; i++) {
        if (!std::isfinite(output[i])) {
            if (++nbad < 100) {
                printf("bad value %f at (row = %zu, col = %zu)\n", output[i], i / ncols, i % ncols);
            }
        }
        err = std::max(err, std::abs(output[i] - outref[i]));
    }
    printf("max absolute error = %.5g, %zu bad values\n", err, nbad);

    // Performance Measuring
    std::vector<double> time_list;
    for (int i = 0; i < 4; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        auto output = SoftmaxCUDA(input, nrows);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;
        time_list.push_back(duration.count());
    }
    double time = *std::min_element(time_list.begin(), time_list.end());
    printf("time = %.4f\n", time);

    return 0;
}
#endif
