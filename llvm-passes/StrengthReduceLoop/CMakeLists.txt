set(LLVM_LINK_COMPONENTS
    Core
    ScalarOpts
    Support
    Analysis
    TransformUtils
    IRReader
)

#add_library(MatMulToNeon)
add_llvm_pass_plugin(StrengthReduceLoop
    StrengthReduceLoop.cpp
)