#pragma once

#include <vector>
#include <cmath>

std::vector<float> Softmax(const std::vector<float>& input, int row_count);

std::vector<float> SoftmaxCUDA(const std::vector<float>& input, int row_count);

std::vector<float> SoftmaxCUDANoPrealloc(const std::vector<float>& input, int row_count);
