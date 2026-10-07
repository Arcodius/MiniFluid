cmake_minimum_required(VERSION 3.20)

project(MiniFluid LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)

add_subdirectory(external/SDL)

file(GLOB SRC_FILES CONFIGURE_DEPENDS src/*.cpp)

add_executable(MiniFluid ${SRC_FILES})

target_link_libraries(MiniFluid PRIVATE SDL3::SDL3)

add_custom_command(TARGET MiniFluid POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    $<TARGET_FILE:SDL3::SDL3>
    $<TARGET_FILE_DIR:MiniFluid>
)

target_compile_options(MiniFluid PRIVATE "/O2") 

option(MINIFLUID_BUILD_CUDA_EXAMPLE "Build the CUDA divergence example" OFF)

if(MINIFLUID_BUILD_CUDA_EXAMPLE)
    enable_language(CUDA)

    add_executable(cuda_divergence_check
        src/cuda/divergence_check.cpp
        src/cuda/divergence_cuda.cu
        src/grid.cpp
        src/math.cpp
    )
    target_include_directories(cuda_divergence_check PRIVATE src)
    set_target_properties(cuda_divergence_check PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
        CUDA_RUNTIME_LIBRARY Shared
    )
endif()