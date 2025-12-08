set(LLVM_LINK_COMPONENTS
    Core
    Support
    Analysis
    TransformUtils
    IRReader
)

#add_library(MatMulToNeon)
add_llvm_pass_plugin(MlirToMatMul
    MlirToMatMul.cpp
)