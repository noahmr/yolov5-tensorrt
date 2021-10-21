cmake_minimum_required(VERSION 3.1)
project(yolov5-tensorrt)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -g -Wall -Wextra -Wno-deprecated -fPIC")
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -g -pthread")

set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})


## Libraries
#####################

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)


## Sources
#####################

include_directories(include)

add_subdirectory(src)


## Examples
#####################

add_subdirectory(examples)
