#include "cuda_check.h"

#include <cstdio>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>

void write_to_log(const std::string& kernel_name, float us) {
    std::ofstream log_file("profile.log", std::ios::app);
    log_file << "[Profile]: " << kernel_name 
             << " | Time: " << us << " μs\n";
    // std::cout << "[LOG] " << kernel_name << " took " << ms << " us\n";
}
