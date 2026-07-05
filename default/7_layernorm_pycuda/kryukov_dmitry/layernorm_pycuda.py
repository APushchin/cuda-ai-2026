import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

KERNEL_CODE = """
__global__ void layernorm_kernel(const float* __restrict__ input,
                                 float* __restrict__ output,
                                 const float* __restrict__ gamma,
                                 const float* __restrict__ beta,
                                 int row_size, float eps)
{
    extern __shared__ float sdata[];

    int row = blockIdx.x;
    int tid = threadIdx.x;
    int row_off = row * row_size;

    float sum = 0.0f;
    for (int i = tid; i < row_size; i += blockDim.x) {
        sum += input[row_off + i];
    }
    sdata[tid] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float mean = sdata[0] / (float)row_size;
    __syncthreads();

    float var_sum = 0.0f;
    for (int i = tid; i < row_size; i += blockDim.x) {
        float diff = input[row_off + i] - mean;
        var_sum += diff * diff;
    }
    sdata[tid] = var_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float inv_std = rsqrtf(sdata[0] / (float)row_size + eps);

    for (int i = tid; i < row_size; i += blockDim.x) {
        float x_hat = (input[row_off + i] - mean) * inv_std;
        output[row_off + i] = gamma[i] * x_hat + beta[i];
    }
}
"""

_mod = SourceModule(KERNEL_CODE)
_kernel = _mod.get_function("layernorm_kernel")

BLOCK_SIZE = 256
SHARED_BYTES = BLOCK_SIZE * 4


class _GpuMem:
    def __init__(self):
        self.d_input = None
        self.d_output = None
        self.d_gamma = None
        self.d_beta = None
        self.input_alloc = 0
        self.gamma_alloc = 0
        self.stream = None

    def ensure(self, input_bytes, gamma_bytes):
        if self.stream is None:
            self.stream = cuda.Stream()

        if input_bytes > self.input_alloc or gamma_bytes > self.gamma_alloc:
            self._free()
            self.d_input = cuda.mem_alloc(input_bytes)
            self.d_output = cuda.mem_alloc(input_bytes)
            self.d_gamma = cuda.mem_alloc(gamma_bytes)
            self.d_beta = cuda.mem_alloc(gamma_bytes)
            self.input_alloc = input_bytes
            self.gamma_alloc = gamma_bytes

    def _free(self):
        for attr in ('d_input', 'd_output', 'd_gamma', 'd_beta'):
            ptr = getattr(self, attr, None)
            if ptr is not None:
                ptr.free()
                setattr(self, attr, None)
        self.input_alloc = 0
        self.gamma_alloc = 0


_mem = _GpuMem()


def layernorm_pycuda(input, gamma, beta, row_size, eps=1e-5):
    inp = np.asarray(input, dtype=np.float32).flatten()
    gam = np.asarray(gamma, dtype=np.float32).flatten()
    bet = np.asarray(beta, dtype=np.float32).flatten()

    total = inp.size
    row_count = total // row_size
    input_bytes = total * 4
    gamma_bytes = row_size * 4

    _mem.ensure(input_bytes, gamma_bytes)

    cuda.memcpy_htod_async(_mem.d_input, inp, _mem.stream)
    cuda.memcpy_htod_async(_mem.d_gamma, gam, _mem.stream)
    cuda.memcpy_htod_async(_mem.d_beta, bet, _mem.stream)

    _kernel(_mem.d_input, _mem.d_output, _mem.d_gamma, _mem.d_beta,
            np.int32(row_size), np.float32(eps),
            block=(BLOCK_SIZE, 1, 1),
            grid=(row_count, 1),
            shared=SHARED_BYTES,
            stream=_mem.stream)

    out = np.empty(total, dtype=np.float32)
    cuda.memcpy_dtoh_async(out, _mem.d_output, _mem.stream)
    _mem.stream.synchronize()

    return out
