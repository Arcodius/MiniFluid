#pragma once

#include <chrono>
#include <cstddef>
#include <string>

void write_to_log(const std::string& kernel_name, float ms);

#define TIME(call) \
    do {  \
        auto start = std::chrono::high_resolution_clock::now();              \
        call;   \
        auto end = std::chrono::high_resolution_clock::now();\
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);\
        write_to_log(#call, static_cast<float>(duration.count()));\
    } while (false)

#define REPLACE(call) call