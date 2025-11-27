#include "qu4d/IR/Qu4dOps.h"
#include "mlir-c/IR.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h" 
#include "mlir/Dialect/Vector/IR/VectorOps.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/LogicalResult.h"
using namespace mlir;
//using namespace mlir::qu4d;

// Generated op definitions (TypeID, parse/print, adaptors, properties, etc.)
#define GET_OP_CLASSES
#include "qu4d/IR/Qu4dOps.cpp.inc"

// Test for memref<4x4xf32> or tensor<4x4xf32> or vector<4xf32>
static bool is4x4f32(Type t) {
    if (auto mt = dyn_cast<MemRefType>(t)) {
        return mt.getRank() == 2 && mt.getDimSize(0) == 4
        && mt.getDimSize(1) == 4 && mt.getElementType().isF32();
    }
    
    if (auto tt = dyn_cast<RankedTensorType>(t)) {
        return tt.getRank() == 2 && tt.getDimSize(0) == 4 
        && tt.getDimSize(1) == 4 && tt.getElementType().isF32();
    }
    return false;
}

static bool isVec4f32(Type t) {
    if (auto vt = dyn_cast<VectorType>(t)) {
        return vt.getElementType().isF32() &&
        vt.getNumElements() == 4;
    }
    return false;
}

static Value stripCastsAndSubViews(Value v) {
    for (;;) {
        if (auto sub = v.getDefiningOp<memref::SubViewOp>()) {
            v = sub.getSource();
            continue;
        }
        if (auto cast = v.getDefiningOp<memref::CastOp>()) {
            v = cast.getSource();
            continue;
        }
        break;
    }
    return v;
}

::llvm::LogicalResult qu4d::Matmul4x4Op::verify() {
    
    Type aTy = getA().getType();
    Type bTy = getB().getType();
    Type cTy = getC().getType();

    Value baseA = stripCastsAndSubViews(getA());
    Value baseB = stripCastsAndSubViews(getB());;
    if (baseA == baseB) {
        return emitOpError("operand A and B should not alias");
    }
    
    if (is4x4f32(aTy) && is4x4f32(bTy) && is4x4f32(cTy)) {
        return llvm::success();;
    }

    if (isVec4f32(aTy) && isVec4f32(bTy) && isVec4f32(cTy)) {
        return llvm::success();
    }

    return emitOpError("operands/results must be memref<4x4xf32>/tensor<4x4xf32> or vector<4xf32>");
}

namespace {
    struct MatMulFoldCanonicalization : mlir::OpRewritePattern<qu4d::Matmul4x4Op> {
        using OpRewritePattern::OpRewritePattern;
        mlir::LogicalResult matchAndRewrite(qu4d::Matmul4x4Op op,
                                            mlir::PatternRewriter &rewriter) const override {
            auto a = op.getA();
            auto b = op.getB();
            auto c = op.getC();
            
            (void)a;
            (void)b; // not folding a and b yet
            
            bool unused_c = true;
            for (auto& use : c.getUses()) {
                if (use.getOwner() != op) {
                    unused_c = false;
                }
            }
            // erase the op is c is never used
            if (unused_c) {
                rewriter.eraseOp(op);
                return mlir::success();
            }

            return rewriter.notifyMatchFailure(
                        op, 
                        "Did not match unused result op");
        } // end matchAndRewrite

    };

} // end anonymous namespace
void qu4d::Matmul4x4Op::getCanonicalizationPatterns(::mlir::RewritePatternSet &patterns, 
                                                    ::mlir::MLIRContext *context) {
    patterns.add<MatMulFoldCanonicalization>(context);
}

/* void qu4d::Matmul4x4Op::build(::mlir::OpBuilder &builder, ::mlir::OperationState &state,
                        ::mlir::Value a, ::mlir::Value b, ::mlir::Value c) {
    state.addOperands({a, b});
    state.addTypes(c.getType());
} */