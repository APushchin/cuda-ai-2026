#include "softmax_cuda.h"

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

namespace {
inline float sec_to_ms(float sec) {
    return sec * 1000.f;
}

template<class T>
std::vector<T> generate_data(size_t size, T range_start = -10, T range_end = 10) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<T> dis(range_start, range_end);
    std::vector<T> data(size);
    std::generate(data.begin(), data.end(), [&]() {
        return dis(gen);
    });
    return data;
}

template<class T>
double mean_abs_diff(const std::vector<T>& a, const std::vector<T>& b) {
    if (a.size() != b.size()) {
        throw std::runtime_error("Size mismatch between vectors");
    }

    double sum = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        sum += std::abs(static_cast<double>(a[i]) - static_cast<double>(b[i]));
    }

    return sum / static_cast<double>(a.size());
}

template<class T>
double max_diff(const std::vector<T>& a, const std::vector<T>& b) {
    if (a.size() != b.size()) {
        throw std::runtime_error("Size mismatch between vectors");
    }

    double max_value = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const double diff = std::abs(
            static_cast<double>(a[i]) - static_cast<double>(b[i])
        );

        max_value = std::max(max_value, diff);
    }

    return max_value;
}

template <typename F, typename... Args>
std::pair<std::vector<float>, double> benchmark(F&& f, Args&&... args) {
    std::vector<double> times;
    std::vector<float> result;

    // warm up
    result = f(args...);

    for (int i = 0; i < 5; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        result = f(args...);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;
        times.push_back(duration.count());
    }
    double time = *std::min_element(times.begin(), times.end());
    return {result, sec_to_ms(time)};
}
}  // namespace

int main(int argc, char* argv[]) {
    size_t n = 1000;

    if (argc > 1) {
        try {
            n = std::stoull(argv[1]);
        } catch (const std::exception& e) {
            std::cerr << "Invalid number of elements: " << argv[1] << '\n';
            return 1;
        }
    }

    size_t elements_num = n * n;
    std::cout << "Elements num (^2): " << elements_num << std::endl;
    std::cout << "------------------------\n";
    std::vector<float> input = generate_data<float>(elements_num);

    auto [ref_res, time_ref] = benchmark(Softmax, input, n);
    std::cout << "Softmax: " << time_ref << " ms" << std::endl;

    auto [res, time] = benchmark(SoftmaxCUDANoPrealloc, input, n);
    std::cout << "Softmax CUDA (no preallocation): " << time << " ms" << std::endl;
    std::cout << "\tMean abs diff: " << std::setprecision(12) <<  mean_abs_diff(ref_res, res) << std::endl;
    std::cout << "\tMax diff: " << std::setprecision(12) <<  max_diff(ref_res, res) << std::endl;

    auto [res_opt, time_opt] = benchmark(SoftmaxCUDA, input, n);
    std::cout << "Softmax CUDA: " << time_opt << " ms" << std::endl;
    std::cout << "\tMean abs diff: " << std::setprecision(12) <<  mean_abs_diff(ref_res, res_opt) << std::endl;
    std::cout << "\tMax diff: " << std::setprecision(12) <<  max_diff(ref_res, res_opt) << std::endl;

    return 0;
}
