#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "qu4d/IR/Qu4dOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Builders.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/Support/LogicalResult.h"
#include <functional>
#include <utility>

using namespace mlir;
using namespace qu4d;
using namespace llvm;
namespace {
struct LowerMatmul4x4Pattern : public OpRewritePattern<qu4d::Matmul4x4Op> {
    LowerMatmul4x4Pattern(MLIRContext *ctx) 
        : OpRewritePattern<Matmul4x4Op>(ctx, 
                                        1, 
                                        Matmul4x4Op::getOperationName()) {}
    
    llvm::LogicalResult matchAndRewrite(Matmul4x4Op op, 
                                        PatternRewriter &rewriter) const override {
        // For memref<4x4xf32>
        Value a = op.getA();
        Value b = op.getB();
        Value c = op.getC();

        auto aTy = dyn_cast<MemRefType>(a.getType());
        auto bTy = dyn_cast<MemRefType>(b.getType());
        auto cTy = dyn_cast<MemRefType>(c.getType());
        if (!aTy || !bTy || !cTy) { 
            return failure();
        }

        if (aTy.getRank() != 2 || bTy.getRank() != 2 || cTy.getRank() != 2) {
            return failure();
        }

        if (!aTy.getElementType().isF32() || !bTy.getElementType().isF32() 
            || !cTy.getElementType().isF32()) {
            return failure();
        }

        Location loc = op->getLoc();
        OpBuilder::InsertionGuard g(rewriter);

        //auto idxType = rewriter.getIndexType();
        Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
        Value four = rewriter.create<arith::ConstantIndexOp>(loc, 4);
        Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);

        // Build: for i in 0..4
        //          for j in 0..4
        //            %sum = 0.0
        //            for k in 0..4
        //              %aval = memref.load a[i,k]
        //              %bval = memref.load b[k,j]
        //              %prod = arith.mulf %aval, %bval
        //              %sum = arith.addf %sum, %prod
        //            memref.store %sum, c[i,j]
        //
        // We use scf.for with step 1.

        // outer loop (i)
/*         auto createLoop = [&](Value lb, Value ub, Value step, 
                            std::function<void(Value)> bodyGen) {
            return rewriter.create<scf::ForOp>(loc, lb, ub, step, ValueRange{}, 
                [&](OpBuilder &b, Location locLoop, Value iv, ValueRange args){
                    bodyGen(iv);
                    b.create<scf::YieldOp>(locLoop);
                }); 

        }; */
        auto iFor = rewriter.create<scf::ForOp>(loc, zero, four, one,
                    ValueRange{},
                    [&](OpBuilder &ib, Location iLoc, Value i, ValueRange /*iterArgs*/) {
                    // j-loop scf.for %j = 0 to 4 step 1
                    auto jFor = ib.create<scf::ForOp> (
                        iLoc, zero, four, one, ValueRange{},
                        [&](OpBuilder &jb, Location jLoc, Value j, ValueRange) {
                        // Initial sum = 0.0
                        auto f32Ty = jb.getF32Type();
                        llvm::APFloat zeroAPF(0.0f);
                        Value initSum = jb.create<arith::ConstantFloatOp>(
                            jLoc, zeroAPF, f32Ty);
                        auto kFor = jb.create<scf::ForOp> (
                            jLoc, zero, four, one, ValueRange{initSum},
                            [&](OpBuilder &kb, Location kLoc,
                            Value k, ValueRange iterArgs) {
                                Value sumIn = iterArgs[0];
                                Value aval = kb.create<memref::LoadOp>(
                                    kLoc, a, ValueRange{i, k}
                                );
                                Value bval = kb.create<memref::LoadOp>(
                                    kLoc, b, ValueRange{k, j}
                                );
                                Value prod = kb.create<arith::MulFOp>(
                                    kLoc, aval, bval
                                );
                                Value sumOut = kb.create<arith::AddFOp>(
                                    kLoc, sumIn, prod
                                );

                                // Pass the new sum to the next iteration
                                kb.create<scf::YieldOp>(kLoc,sumOut);        
                            }
                        ); // end of kFor

                        // Retrieve the result of the k-loop
                        Value finalSum = kFor->getResult(0);

                        jb.create<memref::StoreOp>(jLoc, finalSum, c, 
                                                    ValueRange{i,j});

                        jb.create<scf::YieldOp>(jLoc); // j-loop body terminator
                    });

            // i-loop body terminator
            ib.create<scf::YieldOp>(iLoc);
            (void)jFor;
        }); // end of iFor

        (void)iFor;

        rewriter.eraseOp(op);
        return success();
    }
}; // end struct LowerMatmul4x4Pattern

struct LowerMatmul4x4Pass
    : public PassWrapper<LowerMatmul4x4Pass, OperationPass<ModuleOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerMatmul4x4Pass)

    void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<scf::SCFDialect,
                    arith::ArithDialect,
                    memref::MemRefDialect>();
    }

    StringRef getArgument() const final {
        return "qu4d-lower-matmul4x4-scf";
    }

    StringRef getDescription() const final {
        return "Lower qu4d.matmul_4x4 ops to explicit scf.for loops";
    }
    void runOnOperation() override {
        MLIRContext *ctx = &getContext();
        RewritePatternSet patterns(ctx);
        patterns.add<LowerMatmul4x4Pattern>(ctx);

        if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
            signalPassFailure();
        }
    }
};

} // end anonymous namespace

namespace mlir {
void registerLowerMatMul4x4SCFPass() {
    static PassRegistration<LowerMatmul4x4Pass> reg;
}

} // namespace mlir