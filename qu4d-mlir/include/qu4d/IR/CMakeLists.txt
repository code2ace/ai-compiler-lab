# include/qu4d/IR/CMakeLists.txt
cmake_minimum_required(VERSION 3.16)

# Where source .td files live
set(QU4D_TD_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# Where to put generated headers
set(QU4D_GEN_DIR ${CMAKE_BINARY_DIR}/include/qu4d/IR)
file(MAKE_DIRECTORY ${QU4D_GEN_DIR})

# Try to locate mlir-tblgen. If MLIR provides a variable, prefer that.
if(DEFINED MLIR_TABLEGEN)
  set(MLIR_TBLGEN_EXECUTABLE ${MLIR_TABLEGEN})
else()
  find_program(MLIR_TBLGEN_EXECUTABLE mlir-tblgen HINTS ${MLIR_TOOLS_DIR} PATHS ENV PATH)
endif()

if(NOT MLIR_TBLGEN_EXECUTABLE)
  message(FATAL_ERROR "mlir-tblgen not found. Make sure MLIR build/bin is on PATH or pass -DMLIR_TOOLS_DIR to CMake.")
endif()

message(STATUS "Using mlir-tblgen: ${MLIR_TBLGEN_EXECUTABLE}")

# Determine MLIR include directory (where mlir/IR/DialectBase.td lives).
# Allow user to pass -DMLIR_INCLUDE_DIR; otherwise try reasonable guesses.
if(DEFINED MLIR_INCLUDE_DIR)
  message(STATUS "Using MLIR include dir (from -DMLIR_INCLUDE_DIR): ${MLIR_INCLUDE_DIR}")
else()
  if(DEFINED MLIR_DIR)
    # MLIR_DIR is usually like .../build/lib/cmake/mlir
    get_filename_component(_mlir_cmake_dir_parent ${MLIR_DIR} DIRECTORY) # .../build/lib/cmake
    # Try common relative locations
    set(_guess1 "${_mlir_cmake_dir_parent}/../include")         # e.g. .../build/include
    set(_guess2 "${_mlir_cmake_dir_parent}/../../mlir/include") # e.g. .../llvm-project/mlir/include
    if(EXISTS "${_guess1}")
      set(MLIR_INCLUDE_DIR "${_guess1}")
    elseif(EXISTS "${_guess2}")
      set(MLIR_INCLUDE_DIR "${_guess2}")
    else()
      # last-resort search under the MLIR_DIR parent tree
      find_path(_dialect_base_td NAMES mlir/IR/DialectBase.td HINTS ${_mlir_cmake_dir_parent} PATHS ENV PATH)
      if(_dialect_base_td)
        set(MLIR_INCLUDE_DIR "${_dialect_base_td}")
      endif()
    endif()
  endif()

  if(NOT DEFINED MLIR_INCLUDE_DIR OR NOT EXISTS "${MLIR_INCLUDE_DIR}")
    message(FATAL_ERROR "Could not determine MLIR include directory. Please pass -DMLIR_INCLUDE_DIR=/path/to/mlir/include to cmake.")
  endif()
  message(STATUS "Auto-detected MLIR include dir: ${MLIR_INCLUDE_DIR}")
endif()

# Qu4d ops
set(QU4D_OPS_TD ${QU4D_TD_DIR}/Qu4dOps.td)
set(QU4D_OPS_H_INC ${QU4D_GEN_DIR}/Qu4dOps.h.inc)
set(QU4D_OPS_CPP_INC ${QU4D_GEN_DIR}/Qu4dOps.cpp.inc)

add_custom_command(
  OUTPUT ${QU4D_OPS_H_INC}
  COMMAND ${MLIR_TBLGEN_EXECUTABLE}
          -gen-op-decls
          -I ${QU4D_TD_DIR}
          -I ${MLIR_INCLUDE_DIR}
          -o ${QU4D_OPS_H_INC}
          ${QU4D_OPS_TD}
  DEPENDS ${QU4D_OPS_TD}
  COMMENT "Generating Qu4dOps.h.inc"
)

add_custom_command(
  OUTPUT ${QU4D_OPS_CPP_INC}
  COMMAND ${MLIR_TBLGEN_EXECUTABLE}
          -gen-op-defs
          -I ${QU4D_TD_DIR}
          -I ${MLIR_INCLUDE_DIR}
          -o ${QU4D_OPS_CPP_INC}
          ${QU4D_OPS_TD}
  DEPENDS ${QU4D_OPS_TD}
  COMMENT "Generating Qu4dOps.cpp.inc"
)

# Qu4d dialect
set(QU4D_DIALECT_TD ${QU4D_TD_DIR}/Qu4dDialect.td)
set(QU4D_DIALECT_H_INC ${QU4D_GEN_DIR}/Qu4dDialect.h.inc)
set(QU4D_DIALECT_CPP_INC ${QU4D_GEN_DIR}/Qu4dDialect.cpp.inc)

add_custom_command(
  OUTPUT ${QU4D_DIALECT_H_INC}
  COMMAND ${MLIR_TBLGEN_EXECUTABLE}
          -gen-dialect-decls -dialect=qu4d
          -I ${QU4D_TD_DIR}
          -I ${MLIR_INCLUDE_DIR}
          -o ${QU4D_DIALECT_H_INC}
          ${QU4D_DIALECT_TD}
  DEPENDS ${QU4D_DIALECT_TD}
  COMMENT "Generating Qu4dDialect.h.inc"
)

add_custom_command(
  OUTPUT ${QU4D_DIALECT_CPP_INC}
  COMMAND ${MLIR_TBLGEN_EXECUTABLE}
          -gen-dialect-defs -dialect=qu4d
          -I ${QU4D_TD_DIR}
          -I ${MLIR_INCLUDE_DIR}
          -o ${QU4D_DIALECT_CPP_INC}
          ${QU4D_DIALECT_TD}
  DEPENDS ${QU4D_DIALECT_TD}
  COMMENT "Generating Qu4dDialect.cpp.inc"
)

# Target that depends on all generated files
add_custom_target(Qu4dTableGen
  DEPENDS ${QU4D_OPS_H_INC} ${QU4D_OPS_CPP_INC} ${QU4D_DIALECT_H_INC} ${QU4D_DIALECT_CPP_INC}
)

# Expose include directory so code can #include the generated incs from build tree
include_directories(BEFORE SYSTEM ${QU4D_GEN_DIR})
