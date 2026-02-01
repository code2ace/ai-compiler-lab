add_executable(qu4d-opt qu4d-opt.cpp)
target_link_libraries(qu4d-opt PRIVATE
  MLIRIR 
  MLIRParser
  MLIRSupport

  qu4d-mlir-lib
  MLIRArithDialect 
  MLIRFuncDialect 
  MLIRLinalgDialect
  MLIRMemRefDialect
  MLIRVectorDialect 
  MLIRLLVMDialect 
  MLIRSCFDialect 

  MLIRFuncToLLVM
  MLIRArithToLLVM
  MLIRMemRefToLLVM
  MLIRSCFToControlFlow
  MLIRControlFlowToLLVM
  MLIRVectorToLLVM

  MLIRPass 
  MLIROptLib
  MLIRTransforms
)

# Generated headers live under include/ and build side; include both
target_include_directories(qu4d-opt PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../../include
  ${CMAKE_BINARY_DIR}/include
  ${MLIR_INCLUDE_DIR}
  ${MLIR_BUILD_INCLUDE_DIR}
  ${LLVM_INCLUDE_DIRS}
)