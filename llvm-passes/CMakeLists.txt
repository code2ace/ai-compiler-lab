cmake_minimum_required(VERSION 3.13)
project(llvm-passes)

# Find LLVM (point to your build or install with -DLLVM_DIR=...)
find_package(LLVM REQUIRED CONFIG)

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(AddLLVM)
add_definitions(${LLVM_DEFINITIONS})
include_directories(${LLVM_INCLUDE_DIRS})
link_directories(${LLVM_LIBRARY_DIRS})

# Enable C++17
set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")

add_subdirectory(MatMulToNeon)
add_subdirectory(MlirToMatMul)