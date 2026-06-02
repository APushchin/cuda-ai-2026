#ifndef __GELU_OMP_H
#define __GELU_OMP_H

#include <vector>

std::vector<float> GeluOMP(const std::vector<float>& input);

std::vector<float> GeluSEQ(const std::vector<float>& input);

#endif // __GELU_OMP_H