cmake_minimum_required(VERSION 3.20)
project(MiniFluid LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_subdirectory(external/SDL)

# 1. 递归收集 src 下的所有 C++ 和 CUDA 源文件
file(GLOB_RECURSE ALL_SRC CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu"
)

# 2. 统一构建单条可执行文件
add_executable(MiniFluid ${ALL_SRC})

# 3. 设置头文件路径与库依赖
target_include_directories(MiniFluid PRIVATE src)
target_link_libraries(MiniFluid PRIVATE SDL3::SDL3)
set_target_properties(MiniFluid PROPERTIES CUDA_RUNTIME_LIBRARY Shared)

# 4. 优化选项（MSVC 风格，跨平台可以写成 $<$<CXX_COMPILER_ID:MSVC>:/O2>）
# target_compile_options(MiniFluid PRIVATE $<$<CONFIG:Release,RelWithDebInfo>:/O2>)
# target_link_options(MiniFluid PRIVATE $<$<CONFIG:RelWithDebInfo>:/PROFILE>)

# 5. POST_BUILD 自动化拷贝动态库
add_custom_command(TARGET MiniFluid POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    $<TARGET_FILE:SDL3::SDL3>
    $<TARGET_FILE_DIR:MiniFluid>
)