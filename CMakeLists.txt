cmake_minimum_required(VERSION 3.20)
project(MiniFluid LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_subdirectory(external/SDL)
add_subdirectory(external/DirectXTK12)



# 1. 递归收集 src 下的所有 C++ 和 CUDA 源文件
file(GLOB_RECURSE ALL_SRC CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu"
)

# Dear ImGui：核心 + SDL3 平台后端 + SDL_Renderer 渲染后端
set(IMGUI_DIR ${CMAKE_CURRENT_SOURCE_DIR}/external/imgui)
set(IMGUI_SRC
    ${IMGUI_DIR}/imgui.cpp
    ${IMGUI_DIR}/imgui_draw.cpp
    ${IMGUI_DIR}/imgui_tables.cpp
    ${IMGUI_DIR}/imgui_widgets.cpp
    ${IMGUI_DIR}/backends/imgui_impl_sdl3.cpp
    ${IMGUI_DIR}/backends/imgui_impl_sdlrenderer3.cpp
)

# 2. 统一构建单条可执行文件
add_executable(MiniFluid ${ALL_SRC} ${IMGUI_SRC})

# 3. 设置头文件路径与库依赖
target_include_directories(MiniFluid PRIVATE src ${IMGUI_DIR} ${IMGUI_DIR}/backends)
target_link_libraries(MiniFluid PRIVATE SDL3::SDL3)
target_link_libraries(MiniFluid PRIVATE DirectXTK12)
target_compile_definitions(MiniFluid PRIVATE NOMINMAX)
set_target_properties(MiniFluid PROPERTIES CUDA_RUNTIME_LIBRARY Shared)

# 性能分析器始终编译在内，运行时按 F1 开关。此选项只决定启动时是否默认打开分析器窗口
option(PROFILE "Open the runtime profiler at startup" OFF)
if(PROFILE)
    target_compile_definitions(MiniFluid PRIVATE PROFILE)
endif()

# 4. 优化选项（MSVC 风格，跨平台可以写成 $<$<CXX_COMPILER_ID:MSVC>:/O2>）
# target_compile_options(MiniFluid PRIVATE $<$<CONFIG:Release,RelWithDebInfo>:/O2>)
# target_link_options(MiniFluid PRIVATE $<$<CONFIG:RelWithDebInfo>:/PROFILE>)

# 5. POST_BUILD 自动化拷贝动态库
add_custom_command(TARGET MiniFluid POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    $<TARGET_FILE:SDL3::SDL3>
    $<TARGET_FILE_DIR:MiniFluid>
)

# # main 分支
# cmake --preset release
# cmake --build --preset release

# # cpu-baseline 分支
# cmake --preset release-cpu
# cmake --build --preset release-cpu