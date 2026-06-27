# CUDA Softmax Implementation

### Prerequisites

```
sudo apt install nvidia-cuda-toolkit
```

### Build

```
nvcc main.cu softmax_cuda.cu -o softmax_cuda
```


### Run
```
./softmax_cuda [number of elements]
```

### Results
Results for 100000000 elements:
* Intel(R) Xeon(R) w5-3425, 12 cores,  3.20 GHz
* NVIDIA GeForce RTX 5090

```
Elements num (^2): 100000000
------------------------
Softmax: 1992 ms
Softmax CUDA (no preallocation): 303.561 ms
        Mean abs diff: 6.98108822625e-10
        Max diff: 2.25845724344e-08
Softmax CUDA: 302.32901001 ms
        Mean abs diff: 6.98108822625e-10
        Max diff: 2.25845724344e-08
```
