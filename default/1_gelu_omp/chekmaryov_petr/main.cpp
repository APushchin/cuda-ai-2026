#include <chrono>
#include <stdio.h>
#include <random>
#include <algorithm>
#include "gelu_omp.h"
#include <omp.h>

std::vector<float> GeluOMP_ref(const std::vector<float>& input)
{
    std::vector<float> res(input.size());
    const float* iptr = input.data();
    float* rptr = res.data();
    const float r1 = std::sqrt(2 / M_PI);
    for(int elnum = 0; elnum < (int)input.size(); elnum++)
    {
        float x = iptr[elnum];
        rptr[elnum] = 0.5*x*(1.+std::tanh(r1*(x+0.044715*x*x*x)));
    }
    return res;
}

int main()
{
    omp_set_num_threads(4);
    std::vector<float> input(1000000);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-100.0, 100.0);
    std::generate(input.begin(), input.end(), [&]() { return dist(gen); });
    {
        std::vector<float> out = GeluOMP(input);
        std::vector<float> outref = GeluOMP_ref(input);
        float maxdiff = 0;
        for(int i = 0; i < (int)outref.size();i++) 
            maxdiff = std::abs(outref[i]-out[i]);
        printf("Correctness:%f\n", maxdiff);
    }
    

    std::vector<double> time_list;
    for (int i = 0; i < 100; ++i)
    {
        auto start = std::chrono::high_resolution_clock::now();
        std::vector<float> out = GeluOMP(input);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;
        time_list.push_back(duration.count());
    }
    double time = *std::min_element(time_list.begin(), time_list.end());
    printf("%f\n", time);
}

// Scalar     : 0.011738
// Just OMP   : 0.003385
// OMP + SIMD : 0.002445
//Well, I'm surprised, that compiler didn't made this better, than me.