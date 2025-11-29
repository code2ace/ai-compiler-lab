#include "qu4d/Transforms/Passes.h"

namespace mlir {

void registerQu4dPasses() {
  // Call all your qu4d pass registration functions here
  registerLowerMatMul4x4SCFPass();
  registerQu4dMatmulFromLinalgPass();
}

} // namespace mlir