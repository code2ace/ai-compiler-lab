cmake_minimum_required(VERSION 3.16)
project(qu4d_tests NONE)

# (e.g. ./bin/qu4d-opt <build-root>/test/qu4d/relulowering.mlir ...)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR} DESTINATION ${CMAKE_BINARY_DIR}/test)