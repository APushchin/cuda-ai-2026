#include "softmax_cuda.h"
#include <iostream>
#include <cmath>
#include <cassert>
#include <iomanip>

// Helper function to calculate softmax on CPU for verification
std::vector<float> SoftmaxCPU(const std::vector<float>& input, int row_count) {
    int row_size = input.size() / row_count;
    std::vector<float> output(input.size());
    
    for (int row = 0; row < row_count; row++) {
        // Find max
        float max_val = -INFINITY;
        for (int i = 0; i < row_size; i++) {
            max_val = fmaxf(max_val, input[row * row_size + i]);
        }
        
        // Calculate exp sum
        float sum = 0.0f;
        for (int i = 0; i < row_size; i++) {
            output[row * row_size + i] = expf(input[row * row_size + i] - max_val);
            sum += output[row * row_size + i];
        }
        
        // Normalize
        for (int i = 0; i < row_size; i++) {
            output[row * row_size + i] /= sum;
        }
    }
    
    return output;
}

// Helper to compare two arrays
bool CompareArrays(const std::vector<float>& a, const std::vector<float>& b, float epsilon = 1e-5f) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); i++) {
        if (fabsf(a[i] - b[i]) > epsilon) {
            return false;
        }
    }
    return true;
}

void PrintArray(const std::vector<float>& arr, int row_size, const std::string& name) {
    std::cout << name << ":\n";
    int row_count = arr.size() / row_size;
    for (int row = 0; row < row_count; row++) {
        std::cout << "  Row " << row << ": [";
        for (int i = 0; i < row_size; i++) {
            std::cout << std::fixed << std::setprecision(6) << arr[row * row_size + i];
            if (i < row_size - 1) std::cout << ", ";
        }
        std::cout << "]\n";
    }
}

// TEST 1: Basic single row with 4 elements
void Test1_SingleRow() {
    std::cout << "\n=== Test 1: Single Row ===\n";
    std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> result = SoftmaxCUDA(input, 1);
    std::vector<float> expected = SoftmaxCPU(input, 1);
    
    PrintArray(result, 4, "Result");
    PrintArray(expected, 4, "Expected");
    
    assert(CompareArrays(result, expected));
    std::cout << "PASSED\n";
}

// TEST 2: Multiple rows with equal values
void Test2_MultipleRows() {
    std::cout << "\n=== Test 2: Multiple Rows ===\n";
    std::vector<float> input = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
        7.0f, 8.0f, 9.0f
    };
    std::vector<float> result = SoftmaxCUDA(input, 3);
    std::vector<float> expected = SoftmaxCPU(input, 3);
    
    PrintArray(result, 3, "Result");
    PrintArray(expected, 3, "Expected");
    
    assert(CompareArrays(result, expected));
    std::cout << "PASSED\n";
}

// TEST 3: All identical values (should produce equal probabilities)
void Test3_IdenticalValues() {
    std::cout << "\n=== Test 3: Identical Values ===\n";
    std::vector<float> input = {5.0f, 5.0f, 5.0f, 5.0f, 5.0f};
    std::vector<float> result = SoftmaxCUDA(input, 1);
    
    PrintArray(result, 5, "Result");
    
    // All should be 1/5 = 0.2
    for (float val : result) {
        assert(fabsf(val - 0.2f) < 1e-5f);
    }
    std::cout << "PASSED\n";
}

// TEST 4: Negative values
void Test4_NegativeValues() {
    std::cout << "\n=== Test 4: Negative Values ===\n";
    std::vector<float> input = {-1.0f, -2.0f, -3.0f, -4.0f};
    std::vector<float> result = SoftmaxCUDA(input, 1);
    std::vector<float> expected = SoftmaxCPU(input, 1);
    
    PrintArray(result, 4, "Result");
    
    assert(CompareArrays(result, expected));
    std::cout << "PASSED\n";
}

// TEST 5: Mixed positive and negative values
void Test5_MixedValues() {
    std::cout << "\n=== Test 5: Mixed Positive/Negative ===\n";
    std::vector<float> input = {-2.0f, 0.0f, 2.0f};
    std::vector<float> result = SoftmaxCUDA(input, 1);
    std::vector<float> expected = SoftmaxCPU(input, 1);
    
    PrintArray(result, 3, "Result");
    
    assert(CompareArrays(result, expected));
    std::cout << "PASSED\n";
}

// TEST 6: Very large values (numerical stability)
void Test6_LargeValues() {
    std::cout << "\n=== Test 6: Large Values ===\n";
    std::vector<float> input = {1000.0f, 1001.0f, 1002.0f};
    std::vector<float> result = SoftmaxCUDA(input, 1);
    
    PrintArray(result, 3, "Result");
    
    // Check sum is approximately 1
    float sum = 0.0f;
    for (float val : result) {
        sum += val;
    }
    assert(fabsf(sum - 1.0f) < 1e-5f);
    std::cout << "PASSED\n";
}

// TEST 7: Very small values near zero
void Test7_SmallValues() {
    std::cout << "\n=== Test 7: Small Values ===\n";
    std::vector<float> input = {0.001f, 0.002f, 0.003f};
    std::vector<float> result = SoftmaxCUDA(input, 1);
    std::vector<float> expected = SoftmaxCPU(input, 1);
    
    PrintArray(result, 3, "Result");
    
    assert(CompareArrays(result, expected));
    std::cout << "PASSED\n";
}

// TEST 8: Larger row size (256 elements - matches BLOCK_SIZE)
void Test8_LargeRowSize() {
    std::cout << "\n=== Test 8: Large Row Size (256) ===\n";
    std::vector<float> input(256);
    for (int i = 0; i < 256; i++) {
        input[i] = static_cast<float>(i) / 100.0f;
    }
    
    std::vector<float> result = SoftmaxCUDA(input, 1);
    std::vector<float> expected = SoftmaxCPU(input, 1);
    
    // Check sum is 1
    float sum = 0.0f;
    for (float val : result) {
        sum += val;
    }
    assert(fabsf(sum - 1.0f) < 1e-4f);
    assert(CompareArrays(result, expected, 1e-4f));
    std::cout << "PASSED\n";
}

// TEST 9: Row size larger than BLOCK_SIZE (512 elements)
void Test9_RowSizeLargerThanBlock() {
    std::cout << "\n=== Test 9: Row Size > BLOCK_SIZE (512) ===\n";
    std::vector<float> input(512);
    for (int i = 0; i < 512; i++) {
        input[i] = static_cast<float>(i % 10) / 10.0f;
    }
    
    std::vector<float> result = SoftmaxCUDA(input, 1);
    std::vector<float> expected = SoftmaxCPU(input, 1);
    
    // Check sum is 1
    float sum = 0.0f;
    for (float val : result) {
        sum += val;
    }
    assert(fabsf(sum - 1.0f) < 1e-4f);
    assert(CompareArrays(result, expected, 1e-4f));
    std::cout << "PASSED\n";
}

// TEST 10: Equal values across multiple rows
void Test10_EqualValuesMultipleRows() {
    std::cout << "\n=== Test 10: Equal Values Multiple Rows ===\n";
    std::vector<float> input = {
        1.0f, 1.0f, 1.0f, 1.0f,
        2.0f, 2.0f, 2.0f, 2.0f,
        3.0f, 3.0f, 3.0f, 3.0f
    };
    std::vector<float> result = SoftmaxCUDA(input, 3);
    
    PrintArray(result, 4, "Result");
    
    // Each row should have equal values of 0.25
    for (float val : result) {
        assert(fabsf(val - 0.25f) < 1e-5f);
    }
    std::cout << "PASSED\n";
}

int main() {
    std::cout << "==========================================\n";
    std::cout << "  Running Softmax CUDA Kernel Tests\n";
    std::cout << "==========================================";
    
    try {
        Test1_SingleRow();
        Test2_MultipleRows();
        Test3_IdenticalValues();
        Test4_NegativeValues();
        Test5_MixedValues();
        Test6_LargeValues();
        Test7_SmallValues();
        Test8_LargeRowSize();
        Test9_RowSizeLargerThanBlock();
        Test10_EqualValuesMultipleRows();
        
        std::cout << "\n==========================================\n";
        std::cout << "  ALL TESTS PASSED!\n";
        std::cout << "==========================================\n";
    } catch (const std::exception& e) {
        std::cout << "\nTEST FAILED: " << e.what() << "\n";
        return 1;
    }
    
    return 0;
}