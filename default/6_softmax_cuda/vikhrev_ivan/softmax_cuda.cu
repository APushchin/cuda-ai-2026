#include "softmax_cuda.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <numeric>
#include <cmath>
#include <stdexcept>

static constexpr int BLOCK_SIZE = 16;
static constexpr int REDUCE_BLOCK_SIZE = 256;

std::vector<float> Softmax(const std::vector<float>& input, int row_count) {
    if (input.size() % row_count) {
        throw std::invalid_argument("Input vector size must be divisible by row_count");
    }

    const auto column_count = input.size() / row_count;
    std::vector<float> res(input.size(), 0.f);
    for (auto i = 0; i < row_count; ++i) {
        auto row_start = i * column_count;
        auto row_end = row_start + column_count;

        auto max_value = *std::max_element(input.begin() + row_start, input.begin() +  row_end);
        auto sum_exp = std::accumulate(input.begin() + row_start, input.begin() + row_end, 0.f, [max_value](float sum, float val) {
            return sum + std::exp(val - max_value);
        });

        for (auto j = row_start; j < row_end; ++j) {
            res[j] = std::exp(input[j] - max_value) / sum_exp;
        }
    }

    return res;
}

inline void check_error(cudaError_t ret_code, const std::string& message = "") {
    if (ret_code != cudaSuccess) {
        throw std::runtime_error(std::string(message) + ": " + cudaGetErrorString(ret_code));
    }
}

template <int BLOCK_SIZE>
__global__ void row_max_kernel(const float* input, float* output, int n) {
    int row_id = blockIdx.x;
    int thread_id = threadIdx.x;

    if (row_id >= n) {
        return;
    }

    float local_max = -INFINITY;;
    for (int j = thread_id; j < n; j += blockDim.x) {
        local_max = fmaxf(local_max, input[row_id * n + j]);
    }

    __shared__ float shared_max[BLOCK_SIZE];
    shared_max[threadIdx.x] = local_max;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            shared_max[threadIdx.x] = fmaxf(
                shared_max[threadIdx.x],
                shared_max[threadIdx.x + stride]
            );
        }

        __syncthreads();
    }

    if (threadIdx.x == 0) {
        output[row_id] = shared_max[0];
    }
}


__global__ void exp_shift_kernel(const float* input, float* row_max, float* output, int n) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n && j < n) {
        output[i * n + j] = expf(input[i * n + j] - row_max[i]);
    }
}

template <int BLOCK_SIZE>
__global__ void softmax_kernel(const float* input, float* output, int n) {
    int row_id = blockIdx.x;
    int thread_id = threadIdx.x;

    if (row_id >= n) {
        return;
    }

    float local_sum = 0.f;
    for (int j = thread_id; j < n; j += blockDim.x) {
        local_sum += input[row_id * n + j];
    }

     __shared__ float shared_sum[BLOCK_SIZE];
    shared_sum[threadIdx.x] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + stride];
        }

        __syncthreads();
    }

    for (int j = threadIdx.x; j < n; j += blockDim.x) {
        output[row_id * n + j] = input[row_id * n + j] / shared_sum[0];
    }
    __syncthreads();
}


class CudaFloatBuffer {
public:
    CudaFloatBuffer() = default;

    ~CudaFloatBuffer() {
        cudaFree(d_memory);
    }


    CudaFloatBuffer(const CudaFloatBuffer&) = delete;
    CudaFloatBuffer& operator=(const CudaFloatBuffer&) = delete;

    float* allocate(size_t size) {
        ensure_capacity(size);
        return d_memory;
    }

    float* data() {
        return d_memory;
    }

    const float* data() const {
        return d_memory;
    }

    std::size_t capacity() const {
        return cap;
    }

private:
    void ensure_capacity(std::size_t requested_size) {
        if (requested_size <= cap) {
            return;
        }

        check_error(cudaFree(d_memory), "cudaFree d_memory failed");

        num_bytes = requested_size * sizeof(float);
        check_error(cudaMalloc(&d_memory, num_bytes), "cudaMalloc d_memory failed");

        cap = requested_size;
    }

    std::size_t cap = 0;
    std::size_t num_bytes = 0;

    float* d_memory = nullptr;
};

std::vector<float> SoftmaxCUDA(const std::vector<float>& input, int row_count) {
    static CudaFloatBuffer input_buffer;
    static CudaFloatBuffer row_max_buffer;
    if (input.size() % row_count) {
        throw std::invalid_argument("Input vector size must be divisible by row_count");
    }

    int elements_num = input.size();
    std::vector<float> res(elements_num, 0.f);

    float* d_input = input_buffer.allocate(elements_num);
    float* d_row_max = row_max_buffer.allocate(row_count);

    check_error(cudaMemcpy(
        d_input,
        input.data(),
        elements_num * sizeof(float),
        cudaMemcpyHostToDevice
    ));


    int reduce_blocks = row_count;
    row_max_kernel<REDUCE_BLOCK_SIZE><<<reduce_blocks, REDUCE_BLOCK_SIZE>>>(d_input, d_row_max, row_count);
    check_error(cudaGetLastError());
    check_error(cudaDeviceSynchronize());

    dim3 threads_per_block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks(
        (row_count + threads_per_block.x - 1) / threads_per_block.x,
        (row_count + threads_per_block.y - 1) / threads_per_block.y
    );
    exp_shift_kernel<<<blocks, threads_per_block>>>(d_input, d_row_max, d_input, row_count);
    check_error(cudaGetLastError());
    check_error(cudaDeviceSynchronize());

    softmax_kernel<REDUCE_BLOCK_SIZE><<<reduce_blocks, REDUCE_BLOCK_SIZE>>>(d_input, d_input, row_count);
    check_error(cudaGetLastError());
    check_error(cudaDeviceSynchronize());


    check_error(cudaMemcpy(
        res.data(),
        d_input,
        elements_num * sizeof(float),
        cudaMemcpyDeviceToHost
    ));

    return res;
}

std::vector<float> SoftmaxCUDANoPrealloc(const std::vector<float>& input, int row_count) {
    if (input.size() % row_count) {
        throw std::invalid_argument("Input vector size must be divisible by row_count");
    }

    int elements_num = input.size();
    std::vector<float> res(elements_num, 0.f);

    float* d_input = nullptr;
    float* d_raw_max = nullptr;

    check_error(cudaMalloc(&d_input, elements_num * sizeof(float)));
    check_error(cudaMalloc(&d_raw_max, row_count * sizeof(float)));

    check_error(cudaMemcpy(
        d_input,
        input.data(),
        elements_num * sizeof(float),
        cudaMemcpyHostToDevice
    ));


    int reduce_blocks = row_count;
    row_max_kernel<REDUCE_BLOCK_SIZE><<<reduce_blocks, REDUCE_BLOCK_SIZE>>>(d_input, d_raw_max, row_count);
    check_error(cudaGetLastError());
    check_error(cudaDeviceSynchronize());

    dim3 threads_per_block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks(
        (row_count + threads_per_block.x - 1) / threads_per_block.x,
        (row_count + threads_per_block.y - 1) / threads_per_block.y
    );
    exp_shift_kernel<<<blocks, threads_per_block>>>(d_input, d_raw_max, d_input, row_count);
    check_error(cudaGetLastError());
    check_error(cudaDeviceSynchronize());

    softmax_kernel<REDUCE_BLOCK_SIZE><<<reduce_blocks, REDUCE_BLOCK_SIZE>>>(d_input, d_input, row_count);
    check_error(cudaGetLastError());
    check_error(cudaDeviceSynchronize());


    check_error(cudaMemcpy(
        res.data(),
        d_input,
        elements_num * sizeof(float),
        cudaMemcpyDeviceToHost
    ));

    check_error(cudaFree(d_input));
    check_error(cudaFree(d_raw_max));

    return res;
}
