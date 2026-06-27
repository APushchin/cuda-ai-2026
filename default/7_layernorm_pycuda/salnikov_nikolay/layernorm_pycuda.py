import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

layernorm_kernel = """
__global__ void layernorm_kernel(const float* input, float* output, const float* gamma, const float* beta, int m, int n, float eps) 
{
    int row = blockIdx.x;
    if (row >= m) return;

    int tId = threadIdx.x;
    int bSize = blockDim.x;

    const float* row_in = input + (row * n);
    float* row_out = output + (row * n);

    float sum = 0.0f;
    float sqSum = 0.0f;

    const float4* row_in_vec = reinterpret_cast<const float4*>(row_in);
    int n_vec = n / 4;
    
    for (int j = tId; j < n_vec; j += bSize)
    {
        float4 val = row_in_vec[j];
        
        sum += val.x + val.y + val.z + val.w;
        
        sqSum += (val.x * val.x) + 
                 (val.y * val.y) + 
                 (val.z * val.z) + 
                 (val.w * val.w);
    }

    unsigned int mask = 0xffffffff;
    for (int offset = 16; offset > 0; offset /= 2)
    {
        sum += __shfl_down_sync(mask, sum, offset);
        sqSum += __shfl_down_sync(mask, sqSum, offset);
    }

    __shared__ float s_sum[32];
    __shared__ float s_sqSum[32];

    int lane = tId % 32;
    int wid = tId / 32;

    if (lane == 0)
    {
        s_sum[wid] = sum;
        s_sqSum[wid] = sqSum;
    }
    __syncthreads();

    sum = (tId < (bSize / 32)) ? s_sum[lane] : 0.0f;
    sqSum = (tId < (bSize / 32)) ? s_sqSum[lane] : 0.0f;

    if (wid == 0)
    {
        for (int offset = 16; offset > 0; offset /= 2)
        {
            sum += __shfl_down_sync(mask, sum, offset);
            sqSum += __shfl_down_sync(mask, sqSum, offset);
        }
    }

    __shared__ float s_mean;
    __shared__ float s_inv_std;

    if (tId == 0)
    {
        float mean = sum / n;
        float var = fmaxf(0.0f, (sqSum / n) - (mean * mean)); 
        s_mean = mean;
        s_inv_std = rsqrtf(var + eps);
    }
    __syncthreads();

    float mean = s_mean;
    float inv_std = s_inv_std;

    float4* row_out_vec = reinterpret_cast<float4*>(row_out);
    const float4* gamma_vec = reinterpret_cast<const float4*>(gamma);
    const float4* beta_vec = reinterpret_cast<const float4*>(beta);

    for (int j = tId; j < n_vec; j += bSize)
    {
        float4 val = row_in_vec[j];
        float4 g = gamma_vec[j];
        float4 b = beta_vec[j];
        float4 out;
    
        out.x = g.x * inv_std * (val.x - mean) + b.x;
        out.y = g.y * inv_std * (val.y - mean) + b.y;
        out.z = g.z * inv_std * (val.z - mean) + b.z;
        out.w = g.w * inv_std * (val.w - mean) + b.w;
    
        row_out_vec[j] = out;
    }
    
}
"""

class LayernormStorage:
    def __init__(self):
        self.input_gpu = None
        self.output_gpu = None
        self.gamma_gpu = None
        self.beta_gpu = None
        self.output_cpu = None
        self.input_alloc_bytes = 0
        self.param_alloc_bytes = 0

    def allocate(self, input_len, input_size_bytes, param_size_bytes):
        if input_size_bytes > self.input_alloc_bytes:
            if self.input_gpu is not None:
                self.input_gpu.free()
                self.output_gpu.free()
            self.input_gpu = cuda.mem_alloc(input_size_bytes)
            self.output_gpu = cuda.mem_alloc(input_size_bytes)
            self.output_cpu = np.empty(input_len, dtype=np.float32)
            self.input_alloc_bytes = input_size_bytes

        if param_size_bytes > self.param_alloc_bytes:
            if self.gamma_gpu is not None:
                self.gamma_gpu.free()
                self.beta_gpu.free()
            self.gamma_gpu = cuda.mem_alloc(param_size_bytes)
            self.beta_gpu = cuda.mem_alloc(param_size_bytes)
            self.param_alloc_bytes = param_size_bytes

module = SourceModule(layernorm_kernel, options=["-O3", "-use_fast_math"])
kernel = module.get_function("layernorm_kernel")

storage = LayernormStorage()

def layernorm_pycuda(input, gamma, beta, row_size, eps=1e-5):

    input_cpu = np.ascontiguousarray(input, dtype=np.float32)
    gamma_cpu = np.ascontiguousarray(gamma, dtype=np.float32)
    beta_cpu = np.ascontiguousarray(beta, dtype=np.float32)
  
    m = input_cpu.size // row_size
    n = row_size

    storage.allocate(input_cpu.size, input_cpu.nbytes, gamma_cpu.nbytes)
    
    cuda.memcpy_htod(storage.input_gpu, input_cpu)
    cuda.memcpy_htod(storage.gamma_gpu, gamma_cpu)
    cuda.memcpy_htod(storage.beta_gpu, beta_cpu)
  
    block_dim = 256
    kernel(storage.input_gpu, storage.output_gpu, storage.gamma_gpu, storage.beta_gpu, np.int32(m), np.int32(n), np.float32(eps), block=(block_dim, 1, 1), grid=(m, 1, 1), shared=2048)

    cuda.memcpy_dtoh(storage.output_cpu, storage.output_gpu)
    
    return storage.output_cpu


