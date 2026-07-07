#include "softmax_cuda.h"
#include <cuda_runtime.h>
#include <cmath>

__global__ void softmax_kernel(const float* input,
                               float* output,
                               int row_size,
                               int num_rows) {
    extern __shared__ float sdata[];
    int row = blockIdx.x;
    if (row >= num_rows) return;

    const float* row_in = input + row * row_size;
    float* row_out = output + row * row_size;

    float local_max = -INFINITY;
    for (int i = threadIdx.x; i < row_size; i += blockDim.x) {
        float val = row_in[i];
        if (val > local_max) local_max = val;
    }

    sdata[threadIdx.x] = local_max;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            if (sdata[threadIdx.x + stride] > sdata[threadIdx.x])
                sdata[threadIdx.x] = sdata[threadIdx.x + stride];
        }
        __syncthreads();
    }
    float row_max = sdata[0];
    __syncthreads();

    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < row_size; i += blockDim.x) {
        local_sum += expf(row_in[i] - row_max);
    }

    sdata[threadIdx.x] = local_sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sdata[threadIdx.x] += sdata[threadIdx.x + stride];
        }
        __syncthreads();
    }
    float row_sum = sdata[0];
    __syncthreads();

    for (int i = threadIdx.x; i < row_size; i += blockDim.x) {
        row_out[i] = expf(row_in[i] - row_max) / row_sum;
    }
}

std::vector<float> SoftmaxCUDA(const std::vector<float>& input, int row_count) {
    if (input.empty() || row_count <= 0) return {};

    int row_size = static_cast<int>(input.size() / row_count);
    size_t size = input.size() * sizeof(float);

    float *d_input = nullptr, *d_output = nullptr;
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaMemcpyAsync(d_input, input.data(), size, cudaMemcpyHostToDevice, stream);

    const int threads = 256;
    dim3 grid(row_count);
    dim3 block(threads);
    size_t smem = threads * sizeof(float);
    softmax_kernel<<<grid, block, smem, stream>>>(d_input, d_output, row_size, row_count);

    std::vector<float> result(input.size());
    cudaMemcpyAsync(result.data(), d_output, size, cudaMemcpyDeviceToHost, stream);

    cudaStreamSynchronize(stream);

    cudaStreamDestroy(stream);
    cudaFree(d_input);
    cudaFree(d_output);

    return result;
}