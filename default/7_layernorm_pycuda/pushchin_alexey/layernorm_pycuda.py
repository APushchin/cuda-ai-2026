import numpy as np
import os

import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

FLOAT_TYPE = np.float32
INT_TYPE = np.int32
BLOCK_SIZE = 256
WARP_SIZE = 32

def div_up(a: int, b: int) -> int:
    return (a + b - 1) // b

def layernorm_pycuda(
    input: np.ndarray | list[float],
    gamma: np.ndarray | list[float],
    beta: np.ndarray | list[float],
    row_size: int,
    eps: float = 1e-5,
) -> np.ndarray:
    """
    Apply Layer Normalization to each row of the input matrix.

    Parameters
    ----------
    input : list or numpy.ndarray of float
        Flattened matrix in row‑major order. Its length must be divisible by row_size.
    gamma : list or numpy.ndarray of float
        Scale parameter, length = row_size.
    beta : list or numpy.ndarray of float
        Shift parameter, length = row_size.
    row_size : int
        Number of features per row (i.e., number of columns).
    eps : float, optional
        Small constant for numerical stability.

    Returns
    -------
    numpy.ndarray
        Flattened matrix of the same shape as input, containing the row‑wise
        normalized results.
    """

    input = np.asarray(input, dtype=FLOAT_TYPE)
    gamma = np.asarray(gamma, dtype=FLOAT_TYPE)
    beta = np.asarray(beta, dtype=FLOAT_TYPE)
    eps = eps

    num_columns = row_size
    num_rows = input.size // row_size
    num_warps = BLOCK_SIZE // WARP_SIZE

    cuda_kernels_in_cpp = f"#define BLOCK_SIZE {BLOCK_SIZE}" + """
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
"""
    cuda_kernels_module = SourceModule(cuda_kernels_in_cpp)

    input_device_buffer = cuda.mem_alloc(input.nbytes)
    gamma_device_buffer = cuda.mem_alloc(gamma.nbytes)
    beta_device_buffer = cuda.mem_alloc(beta.nbytes)
    output_device_buffer = cuda.mem_alloc(input.nbytes)

    cuda.memcpy_htod(input_device_buffer, input)
    cuda.memcpy_htod(gamma_device_buffer, gamma)
    cuda.memcpy_htod(beta_device_buffer, beta)

    cuda_kernels_module.get_function("layernorm_kernel")(
        output_device_buffer,
        input_device_buffer,
        gamma_device_buffer,
        beta_device_buffer,
        FLOAT_TYPE(eps),
        INT_TYPE(num_rows),
        INT_TYPE(num_columns),
        block=(BLOCK_SIZE, 1, 1),
        grid=(num_rows, 1, 1),
    )

    result = np.empty_like(input)
    cuda.memcpy_dtoh(result, output_device_buffer)

    input_device_buffer.free()
    gamma_device_buffer.free()
    beta_device_buffer.free()
    output_device_buffer.free()

    return result

if __name__ == "__main__":
    import random
    from timeit import default_timer as timer

    def layernorm_reference(
        input: np.ndarray,
        gamma: np.ndarray,
        beta: np.ndarray,
        eps: FLOAT_TYPE,
    ) -> np.ndarray:
        mean = np.mean(input, axis=-1, keepdims=True)
        variance = np.var(input, axis=-1, keepdims=True)
        input_normalized = (input - mean) / np.sqrt(variance + eps)
        # print(f"{input=}\n{mean=}\n{variance=}\n{input_normalized=}\n{gamma=}\n{beta=}")
        return gamma * input_normalized + beta

    class RandomGenerator:
        def __init__(self):
            self.rng = np.random.default_rng(seed=42)
            random.seed(42)

        def generate_random_vector(self, size: int) -> np.array:
            return self.rng.normal(
                loc=random.random(),
                scale=random.random(),
                size=size,
            ).astype(FLOAT_TYPE)

    num_experiments = 5
    num_rows = 8192
    num_columns = 8192

    random_generator = RandomGenerator()
    input = np.zeros((num_rows, num_columns), dtype=FLOAT_TYPE)
    for row_idx in range(num_rows):
        input[row_idx, :] = random_generator.generate_random_vector(num_columns)
    gamma = random_generator.generate_random_vector(num_columns)
    beta = random_generator.generate_random_vector(num_columns)
    eps = 1e-5

    result_reference = layernorm_reference(input, gamma, beta, FLOAT_TYPE(eps)).ravel()
    input = input.ravel()

    _ = layernorm_pycuda(input, gamma, beta, num_columns, eps) # warming up

    latencies = []
    max_absolute_error = 0
    max_relative_error = 0
    for experiment_id in range(num_experiments):
        start = timer()
        result = layernorm_pycuda(input, gamma, beta, num_columns, eps)
        latencies.append(timer() - start)
        error = result_reference - result
        max_absolute_error = max(max_absolute_error, np.abs(error).max())
        max_relative_error = max(max_relative_error, np.abs(error / result_reference).max())
        errors_are_large = np.abs(error / result_reference) > 0.001
        print(f"{np.sum(errors_are_large)=}")
        # print(f"{np.argwhere(errors_are_large)=}")
        # print(f"{result_reference=}\n{result=}")

    total_time = sum(latencies)
    min_time = min(latencies)
    print(
        f"Taking Layernorm from {num_columns}x{num_rows} matrix "
        f"{num_experiments} times took {total_time} seconds"
        f": mean={total_time / num_experiments}s"
        f", min={min_time}s"
        f"\nMax errors: absolute={max_absolute_error}"
        f", relative={max_relative_error}"
    )
