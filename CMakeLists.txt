cmake_minimum_required(VERSION 3.20)

project(MiniFluid LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

add_subdirectory(external/SDL)

add_executable(MiniFluid
    src/main.cpp
)

target_link_libraries(MiniFluid PRIVATE SDL3::SDL3)

add_custom_command(TARGET MiniFluid POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    $<TARGET_FILE:SDL3::SDL3>
    $<TARGET_FILE_DIR:MiniFluid>
)