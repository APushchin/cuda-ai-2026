#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <random>
#include <algorithm>
#include <iomanip>
#include "gelu_omp.h"

// Функция для генерации тестовых данных
std::vector<float> generate_data(size_t size) {
    std::vector<float> data(size);
    std::mt19937 gen(42); // Фиксированный seed для воспроизводимости
    // GELU наиболее интересна в диапазоне от -4 до 4
    std::uniform_real_distribution<float> dis(-6.0f, 6.0f); 
    
    for (size_t i = 0; i < size; ++i) {
        data[i] = dis(gen);
    }
    return data;
}

int main() {
    // 10 миллионов элементов для надежного замера времени
    const size_t DATA_SIZE = 134217728; 
    const float EPSILON = 0.03f; // Порог 3%
    
    std::cout << "Генерация " << DATA_SIZE << " элементов..." << std::endl;
    auto input = generate_data(DATA_SIZE);

    // --- ЗАМЕР СКОРОСТИ GELUREF ---
    std::cout << "Запуск GeluRef..." << std::endl;
    auto start_ref = std::chrono::high_resolution_clock::now();
    auto res_ref = GeluRef(input);
    auto end_ref = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration_ref = end_ref - start_ref;

    // --- ЗАМЕР СКОРОСТИ GELUOMP ---
    std::cout << "Запуск GeluOMP..." << std::endl;
    auto start_omp = std::chrono::high_resolution_clock::now();
    auto res_omp = GeluOMP(input);
    auto end_omp = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration_omp = end_omp - start_omp;

    // --- ПРОВЕРКА ТОЧНОСТИ ---
    bool precision_passed = true;
    size_t error_count = 0;
    float max_rel_error = 0.0f;

    if (res_ref.size() != res_omp.size()) {
        std::cerr << "Ошибка: Размеры выходных векторов не совпадают!" << std::endl;
        return 1;
    }

    for (size_t i = 0; i < res_ref.size(); ++i) {
        float ref_val = res_ref[i];
        float omp_val = res_omp[i];
        
        float abs_error = std::abs(ref_val - omp_val);
        float rel_error = 0.0f;

        // Защита от деления на 0 при расчете относительной погрешности
        if (std::abs(ref_val) > 1e-5f) {
            rel_error = abs_error / std::abs(ref_val);
        } else {
            // Если эталон около нуля, проверяем абсолютную ошибку по тому же порогу
            rel_error = abs_error; 
        }

        if (rel_error > max_rel_error) {
            max_rel_error = rel_error;
        }

        if (rel_error > EPSILON) {
            precision_passed = false;
            error_count++;
            // Выведем первые несколько ошибок для отладки
            if (error_count <= 5) {
                std::cout << "  [Ошибка] Индекс " << i << ", Вход: " << input[i]
                          << ", Ref: " << ref_val << ", OMP: " << omp_val 
                          << ", Относ. погрешность: " << rel_error * 100 << "%" << std::endl;
            }
        }
    }

    // --- ВЫВОД РЕЗУЛЬТАТОВ ---
    std::cout << "\n================ РЕЗУЛЬТАТЫ ================" << std::endl;
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Время работы GeluRef: " << duration_ref.count() << " мс" << std::endl;
    std::cout << "Время работы GeluOMP: " << duration_omp.count() << " мс" << std::endl;
    
    if (duration_omp.count() > 0) {
        std::cout << "Ускорение OMP:        " << (duration_ref.count() / duration_omp.count()) << "x" << std::endl;
    }

    std::cout << "--------------------------------------------" << std::endl;
    std::cout << "Макс. зафиксированная погрешность: " << (max_rel_error * 100) << "%" << std::endl;
    
    if (precision_passed) {
        std::cout << "Проверка точности:    УСПЕШНО (все погрешности < " << (EPSILON * 100) << "%)" << std::endl;
    } else {
        std::cout << "Проверка точности:    НЕ ПРОЙДЕНА (" << error_count << " ошибок из " << DATA_SIZE << ")" << std::endl;
    }
    std::cout << "============================================" << std::endl;

    return 0;
}
