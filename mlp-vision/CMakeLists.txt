cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
project(mlp-vision)

# EZ A LÉNYEG: Megmondjuk neki, hol a LibTorch a te gépeden
set(CMAKE_PREFIX_PATH "/Users/arintonakos/Developer/libs/libtorch")

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(mlp-vision main.cpp)
target_link_libraries(mlp-vision "${TORCH_LIBRARIES}")
set_property(TARGET mlp-vision PROPERTY CXX_STANDARD 17)