#include "qu4d/IR/Qu4dDialect.h"
#include "qu4d/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

int main(int argc, char **argv) {
  DialectRegistry registry;
  registry.insert<mlir::func::FuncDialect,
                  scf::SCFDialect,
                  arith::ArithDialect,
                  memref::MemRefDialect,
                  qu4d::Qu4dDialect>();

  // Register all passes defined in your qu4d project
  registerQu4dPasses();
  mlir::registerCanonicalizerPass(); 

  return failed(MlirOptMain(
      argc, argv,
      "Qu4d optimizer driver\n",
      registry));
    // or this
/*       return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Qu4d optimizer\n", registry)); */
}