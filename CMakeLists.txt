cmake_minimum_required (VERSION 3.1)

project(network)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_BUILD_TYPE Release)

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O2")

include_directories(include)

add_library(mlp SHARED src/network.cpp src/layer.cpp src/layer_deque.cpp src/transformation.cpp)
