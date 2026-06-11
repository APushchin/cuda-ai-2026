#include "softmax_cuda.h"

#include <float.h>
#include <math.h>
#include <iostream>
#include <vector>

template <int BLOCK_SIZE = 512>
__global__ void softmax_kernel(const float* __restrict__ input,
                                       float* __restrict__ output,
                                       int N, int D) {
    int row = blockIdx.x;
    if (row >= N) return;

    extern __shared__ float shmem[];
    float* exps  = shmem;
    float* red   = shmem + D;

    int tid = threadIdx.x;
    float local_max = -FLT_MAX;

    for (int i = tid; i < D; i += BLOCK_SIZE) {
        local_max = fmaxf(local_max, __ldg(&input[row * D + i]));
    }

    red[tid] = local_max;
    __syncthreads();
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            red[tid] = fmaxf(red[tid], red[tid + stride]);
        }
        __syncthreads();
    }
    float row_max = red[0];

    for (int i = tid; i < D; i += BLOCK_SIZE) {
        exps[i] = expf(__ldg(&input[row * D + i]) - row_max);
    }

    red[tid] = (tid < D) ? exps[tid] : 0.0f;
    __syncthreads();
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            red[tid] += red[tid + stride];
        }
        __syncthreads();
    }
    float row_sum = red[0];

    for (int i = tid; i < D; i += BLOCK_SIZE) {
        output[row * D + i] = exps[i] / row_sum;
    }
}

void softmax(const float* d_input, float* d_output, int N, int D, cudaStream_t stream = 0)
{
    const int blockSize = 512;
    int sharedMemSize = (D + blockSize) * sizeof(float);

    dim3 grid(N);
    dim3 block(blockSize);

    softmax_kernel<blockSize><<<grid, block, sharedMemSize, stream>>>(d_input, d_output, N, D);
}

std::vector<float> SoftmaxCUDA(const std::vector<float>& input, int row_count)
{
    const int N = row_count;
    const int D = static_cast<int>(input.size()) / row_count;

    std::vector<float> h_output(N*D);

    float *d_in, *d_out;
    cudaMalloc(&d_in,  N * D * sizeof(float));
    cudaMalloc(&d_out, N * D * sizeof(float));
    cudaMemcpy(d_in, input.data(), N * D * sizeof(float), cudaMemcpyHostToDevice);

    softmax(d_in, d_out, N, D);

    cudaMemcpy(h_output.data(), d_out, N * D * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_in);
    cudaFree(d_out);

    return h_output;
}

// int main()
// {
//     int N = 16;  // batch size
//     int D = 16;   // features per sample

//     std::vector<float> h_input(N*D, 0.0), h_output(N*D, 0.0);

//     for(int i = 0; i < N; i++)
//         for(int j = 0; j < N; j++)
//         {
//             h_input[i*N+j] = i+j+2;
//              std::cout << i*N+j << ", "<< h_input[i*N+j] << std::endl;
//         }

//     h_output = SoftmaxCUDA(h_input, N);

//     for(int i = 0; i < N; i++)
//     {
//         for(int j = 0; j < N; j++)
//         {
//             std::cout << h_output[i*N+j] << std::endl;
//         }
//     }

//     return 0;
// }
