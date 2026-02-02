#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "qu4d/IR/Qu4dOps.h"
#include "qu4d/Transforms/Passes.h"
#include "llvm/Support/Casting.h"
#include <utility>

using namespace mlir;
using namespace qu4d;
using namespace llvm;

namespace {
struct LinalgMatmulToQu4dPattern
    : public ::mlir::OpRewritePattern<mlir::linalg::MatmulOp> {
    using OpRewritePattern::OpRewritePattern;
    mlir::LogicalResult matchAndRewrite(mlir::linalg::MatmulOp op,
                                        mlir::PatternRewriter &rewriter) const override {
       /*  Value a = op.getOperand(0);
        Value b = op.getOperand(1);
        Value c = op->getOperand(2); */
        Value a = op.getInputs()[0];
        Value b = op.getInputs()[1];
        Value c = op.getOutputs()[0];

        auto aTy = dyn_cast<MemRefType>(a.getType());
        auto bTy = dyn_cast<MemRefType>(b.getType());
        auto cTy = dyn_cast<MemRefType>(c.getType());
    
        if (!aTy || !bTy || !cTy) {
            return rewriter.notifyMatchFailure(op, 
                "Input/output operands are NOT all memrefs");
        }

        auto is4x4F32Type = [](MemRefType ty) {
            return ty.getRank() == 2 
                && ty.getShape()[0] == 4 
                && ty.getShape()[1] == 4
                && ty.getElementType().isF32();
        };

        if (!is4x4F32Type(aTy) || !is4x4F32Type(bTy) 
            || !is4x4F32Type(cTy)) {
            return rewriter.notifyMatchFailure(op, 
                "Not a 4x4xf32 memref matmul");
        }

        qu4d::Matmul4x4Op matmulOp = rewriter.create<qu4d::Matmul4x4Op>(op->getLoc(), a, b, c);
        rewriter.eraseOp(op);
        return success();
    }

}; // end struct LinalgMatmulToQu4dPattern

struct Qu4dMatmulFromLinalgPass 
    : public ::mlir::PassWrapper<Qu4dMatmulFromLinalgPass, 
                          ::mlir::OperationPass<mlir::func::FuncOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(Qu4dMatmulFromLinalgPass)
    
    void getDependentDialects(mlir::DialectRegistry &registry) const override {
        registry.insert<
            mlir::linalg::LinalgDialect,
            mlir::memref::MemRefDialect,
            qu4d::Qu4dDialect>();
    }

    void runOnOperation() override;

    StringRef getArgument() const final {
        return "qu4d-from-linalg";
    }

    StringRef getDescription() const final {
        return "Convert 4x4 linalg.matmul on memrefs to qu4d.matmul_4x4";
    }
}; // end struct Qu4dMatmulFromLinalgPass

void Qu4dMatmulFromLinalgPass::runOnOperation() {
    auto func = getOperation();
    MLIRContext *ctx = func->getContext();

    RewritePatternSet patterns(ctx);
    patterns.add<LinalgMatmulToQu4dPattern>(ctx);
    if (failed(applyPatternsGreedily(func, std::move(patterns)))) {
        signalPassFailure();
    }
}
} // end anonymous namespace

// Register the pass
namespace mlir {
void registerQu4dMatmulFromLinalgPass() {
    static PassRegistration<Qu4dMatmulFromLinalgPass> reg;
}
}
