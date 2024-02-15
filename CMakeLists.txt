cmake_minimum_required (VERSION 3.1)

project(network)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_BUILD_TYPE "Release")

include_directories(include)

add_library(mlp SHARED src/drawer.cpp src/network.cpp src/bnetwork.cpp src/cnetwork.cpp src/gnetwork.cpp
    src/layer.cpp src/blayer.cpp src/glayer.cpp src/layer_deque.cpp src/transformation.cpp src/kmeans.cpp)
