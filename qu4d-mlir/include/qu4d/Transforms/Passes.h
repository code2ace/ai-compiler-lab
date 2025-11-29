#pragma once

//#include "mlir/Pass/Pass.h"

namespace mlir {

// Project-wide registration entry point
void registerQu4dPasses();

// Specific pass registration (already implemented in LowerMatmul4x4.cpp)
void registerLowerMatMul4x4SCFPass();
void registerQu4dMatmulFromLinalgPass();

} // namespace mlir