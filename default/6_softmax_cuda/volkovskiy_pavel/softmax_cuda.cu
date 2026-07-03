#include "softmax_cuda.h"

#include <cuda/cmath>
#include <cuda_runtime.h>

#include <thread>

static constexpr float minus_infinity = -std::numeric_limits<float>::infinity(); 

static constexpr unsigned WARP_SIZE = 32;

__global__ void softmax_kernel(const float* in, float* out, size_t n, size_t nrow, size_t ncol)
{
    const int tcol = threadIdx.x;
    const int r = blockIdx.y * blockDim.y + threadIdx.y;
    const int stride = blockDim.x;
    extern __shared__ float buffer[];
    float* shared_max = buffer;
    float* shared_sum = &buffer[blockDim.x * blockDim.y];

    if (r < nrow) {
        const float *in_row = &in[ncol * r];
        float *out_row = &out[ncol * r];
        float max = minus_infinity;
        for (int i = tcol; i < ncol; i += stride) {
            float val = in_row[i];
            max = fmaxf(max, val);
        }

        shared_max[blockDim.x  * threadIdx.y  + threadIdx.x] = max;

        __syncthreads();

        max = minus_infinity;
        for (int i = 0; i < stride; ++i) {
            float val = shared_max[blockDim.x  * threadIdx.y  + i];
            max = fmaxf(max, val);
        }
        float sum = 0.0f;
        for (int i = tcol; i < ncol; i += stride) {
            float val = expf(in_row[i] - max);
            sum += val;
            out_row[i] = val;
        }
        shared_sum[blockDim.x  * threadIdx.y  + threadIdx.x] = sum;

        __syncthreads();

        sum = 0.0f;
        for (int i = 0; i < stride; ++i) {
            float val = shared_sum[blockDim.x  * threadIdx.y  + i];
            sum += val;
        }

        float scale = 1.0f / sum;
        for (int i = tcol; i < ncol; i += stride) {
            out_row[i] *= scale;
        }
    } // if r < nrow
}

class cuda_memory_buffer
{
public:

    cuda_memory_buffer(const std::size_t bytes = 0) {
        allocate(bytes);
    }
    cuda_memory_buffer(cuda_memory_buffer&& other) = delete;

    ~cuda_memory_buffer()
    {
        freebuf();
    }

    void* allocate(const std::size_t bytes) {
        if (bytes > sz ) {
            freebuf();
            cudaError_t st = cudaMalloc((void**)&buf, bytes);
            if (cudaSuccess == st) {
                sz = bytes;
                return buf;
            } else {
                sz = 0;
                buf = nullptr;
            }

        }
        return buf;
    }

    void freebuf() {
        if (buf) {
            cudaFree(buf);
            buf = nullptr;
            sz = 0;
        }
    }

private:
    void *buf = nullptr;
    std::size_t sz = 0;
};

std::vector<float> SoftmaxCUDA(const std::vector<float>& input, int row_count)
{
    const std::size_t input_size = input.size();
    const std::size_t col_count = input_size / row_count;
    const std::size_t memsize = input_size * sizeof(float);
    
    std::vector<float> result;
    
    std::thread t([&result, input_size]() {
        result.resize(input_size);
    });

    static int minGridSize = 0;
    static int maxBlockSize = 0;
    static bool isBlockSizeComputed = false;
    if (!isBlockSizeComputed) {
        isBlockSizeComputed = true;
        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &maxBlockSize, softmax_kernel);
    }

    dim3 threadsPerBlock(WARP_SIZE, maxBlockSize / WARP_SIZE);
    dim3 numBlocks(1, cuda::ceil_div((unsigned)row_count, threadsPerBlock.y));
    const std::size_t sharedCacheSize = 2 * maxBlockSize * sizeof(float);

    static cuda_memory_buffer inputbuf;
    static cuda_memory_buffer outputbuf;

    float *in = static_cast<float*>(inputbuf.allocate(memsize));
    float *out =  static_cast<float*>(outputbuf.allocate(memsize));


    cudaMemcpy(in, input.data(), memsize, cudaMemcpyHostToDevice);

    softmax_kernel<<<numBlocks, threadsPerBlock, sharedCacheSize >>>(in, out, input_size, row_count, col_count);

    t.join();
    cudaMemcpy(result.data(), out, memsize, cudaMemcpyDeviceToHost);

    return result;
}


