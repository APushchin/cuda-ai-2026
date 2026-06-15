#include <chrono>
#include <iostream>
#include <random>
#include <vector>

#include "naive_gemm_cuda.h"


std::vector<float> NaiveGemmRef(const std::vector<float>& a, const std::vector<float>& b, int n) {
    if (a.size() != n * n || b.size() != n * n) {
        throw std::invalid_argument("Matrix dimensions do not match vector sizes.");
    }

    std::vector<float> c(n * n, 0.0f);

    for (int i = 0; i < n; ++i) {
        for (int k = 0; k < n; ++k) {
            for (int j = 0; j < n; ++j) {
                c[i * n + j] += a[i * n + k] * b[k * n + j];
            }
        }
    }

    return c;
}

// Функция для генерации случайной матрицы
std::vector<float> GenerateRandomMatrix(int rows, int cols) {
    std::vector<float> matrix(rows * cols);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    
    for (int i = 0; i < rows * cols; ++i) {
        matrix[i] = dis(gen);
    }
    return matrix;
}

// Функция для сравнения точности двух матриц
float ComputeMaxAbsoluteError(const std::vector<float>& ref, const std::vector<float>& test) {
    float max_error = 0.0f;
    for (size_t i = 0; i < ref.size(); ++i) {
        float error = std::abs(ref[i] - test[i]);
        if (error > max_error) {
            max_error = error;
        }
    }
    return max_error;
}

int main() {
    // Размер матрицы N x N
    const int N = 4096; 
    std::cout << "Матрицы размером: " << N << "x" << N << "\n\n";

    // Инициализация данных
    auto a = GenerateRandomMatrix(N, N);
    auto b = GenerateRandomMatrix(N, N);

    // 1. Тестирование NaiveGemmRef
    std::cout << "Запуск NaiveGemmRef (CPU)..." << std::endl;
    auto start_ref = std::chrono::high_resolution_clock::now();
    
    auto res_ref = NaiveGemmRef(a, b, N);
    
    auto end_ref = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration_ref = end_ref - start_ref;
    std::cout << "Время CPU: " << duration_ref.count() << " мс\n\n";

    // 2. Тестирование NaiveGemmCUDA
    std::cout << "Запуск NaiveGemmCUDA (GPU)..." << std::endl;
    // Первая итерация часто включает инициализацию контекста CUDA, делаем "разогрев" при необходимости
    auto start_cuda = std::chrono::high_resolution_clock::now();
    
    auto res_cuda = NaiveGemmCUDA(a, b, N);
    
    auto end_cuda = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration_cuda = end_cuda - start_cuda;
    std::cout << "Время CUDA: " << duration_cuda.count() << " мс\n\n";

    // 3. Проверка точности
    float max_error = ComputeMaxAbsoluteError(res_ref, res_cuda);
    std::cout << "=== Результаты сравнения ===" << std::endl;
    std::cout << "Максимальная абсолютная ошибка: " << max_error << std::endl;
    
    // Допустимый порог ошибки из-за погрешности float (около 1e-4 или 1e-5 в зависимости от N)
    const float epsilon = 1e-4f;
    if (max_error < epsilon) {
        std::cout << "СТАТУС: УСПЕШНО (Результаты совпадают)" << std::endl;
    } else {
        std::cout << "СТАТУС: ОШИБКА (Слишком большая разница в вычислениях!)" << std::endl;
    }

    // Ускорение
    double speedup = duration_ref.count() / duration_cuda.count();
    std::cout << "Ускорение CUDA относительно CPU: " << speedup << " x" << std::endl;

    return 0;
}
