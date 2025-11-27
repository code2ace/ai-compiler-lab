add_library(qu4d-mlir-lib
  IR/Qu4dDialect.cpp
  IR/Qu4dOps.cpp
  Transforms/LowerToArith.cpp
  Transforms/DetectMicrokernel.cpp
  Transforms/LowerMatmul4x4.cpp
  Transforms/Passes.cpp
)

set_target_properties(qu4d-mlir-lib PROPERTIES OUTPUT_NAME "qu4d-mlir")

# Link MLIR bits you touch
target_link_libraries(qu4d-mlir-lib PRIVATE
  MLIRIR MLIRArithDialect MLIRFuncDialect MLIRLinalgDialect MLIRVectorDialect
)

# --- compute MLIR build include dir (where generated .inc files live) ---
if(DEFINED MLIR_DIR)
  # MLIR_DIR is typically: <build-root>/lib/cmake/mlir
  # We try a few guesses relative to MLIR_DIR; the most common is:
  #   <build-root>/tools/mlir/include
  get_filename_component(_mlir_cmake_dir_parent ${MLIR_DIR} DIRECTORY) # -> <build-root>/lib/cmake
  set(_guess_tools_mlir "${_mlir_cmake_dir_parent}/../../tools/mlir/include")
  set(_guess_build_include "${_mlir_cmake_dir_parent}/../include")  # -> <build-root>/include
  if(EXISTS "${_guess_tools_mlir}")
    set(MLIR_BUILD_INCLUDE_DIR "${_guess_tools_mlir}")
  elseif(EXISTS "${_guess_build_include}")
    set(MLIR_BUILD_INCLUDE_DIR "${_guess_build_include}")
  else()
    # fallback: let users override with -DMLIR_BUILD_INCLUDE_DIR=...
    if(NOT DEFINED MLIR_BUILD_INCLUDE_DIR)
      message(WARNING "Could not auto-detect MLIR build include dir; if compile errors occur, re-run cmake with -DMLIR_BUILD_INCLUDE_DIR=/path/to/build/tools/mlir/include")
    endif()
  endif()
endif()

target_include_directories(qu4d-mlir-lib PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../include
  ${CMAKE_BINARY_DIR}/include
  ${MLIR_INCLUDE_DIR}                         # e.g. .../llvm-project/mlir/include
  ${MLIR_BUILD_INCLUDE_DIR}
  ${LLVM_INCLUDE_DIRS}                        # llvm headers (already found by find_package)
)

# Ensure TableGen runs before compiling
add_dependencies(qu4d-mlir-lib Qu4dTableGen)
