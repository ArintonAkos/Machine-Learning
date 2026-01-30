cmake_minimum_required(VERSION 3.10)
project(AppSiliconResize)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Default to Release build type for performance
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler optimization flags
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    add_compile_options(-O3)
endif()

# Option to toggle SIMD (Default: ON)
option(ENABLE_SIMD "Enable SIMD instructions" ON)

if(NOT ENABLE_SIMD)
    add_compile_definitions(STBIR_NO_SIMD)
    message(STATUS "SIMD: DISABLED (Scalar Mode)")
else()
    message(STATUS "SIMD: ENABLED (Library Default / NEON)")
endif()

add_executable(benchmark main.cpp)
