__global__ void layernorm_kernel(
    float* output,
    const float* input,
    const float* gamma,
    const float* beta,
    float epsilon,
    int num_rows,
    int num_columns
) {
    const auto row_idx = blockIdx.x;
    if (row_idx >= num_rows) return;

    const auto thread_idx = threadIdx.x;

    __shared__ float shared_sum[BLOCK_SIZE];
    __shared__ float mean_across_row;
    __shared__ float variance_across_row;

    auto sum = 0.f;
    for (int i = thread_idx; i < num_columns; i += blockDim.x) {
        sum += input[row_idx * num_columns + i];
    }
    shared_sum[threadIdx.x] = sum;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (thread_idx == 0) {
        mean_across_row = shared_sum[0] / num_columns;
    }
    __syncthreads();

    float variance_sum = 0.f;
    for (int i = thread_idx; i < num_columns; i += blockDim.x) {
        const auto difference = input[row_idx * num_columns + i] - mean_across_row;
        variance_sum += difference * difference;
    }

    shared_sum[thread_idx] = variance_sum;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (thread_idx < stride) {
            shared_sum[thread_idx] += shared_sum[thread_idx + stride];
        }
        __syncthreads();
    }

    if (thread_idx == 0) {
        variance_across_row = shared_sum[0] / num_columns;
    }
    __syncthreads();

    float inversed_standard_deviance = rsqrtf(variance_across_row + epsilon);
    for (int i = threadIdx.x; i < num_columns; i += blockDim.x) {
        const auto idx = row_idx * num_columns + i;
        output[idx] = (
            gamma[i] * (input[idx] - mean_across_row) * inversed_standard_deviance + beta[i]
        );
    }
    __syncthreads();
}
