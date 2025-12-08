#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/IR/Analysis.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Value.h"
#include "llvm/Pass.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/LoopUtils.h"
#include <optional>

using namespace llvm;
struct CanonicalLoop {
    Loop *L;
    PHINode *IndVar;
    uint64_t TripCount;
/*     BasicBlock *Header;
    BasicBlock *Latch;
    BasicBlock *Preheader;
    BasicBlock *Exit; */
};

struct MatMulNest {
    CanonicalLoop Outer;
    CanonicalLoop Middle;
    CanonicalLoop Inner;
};

struct MatMulBase {
    Value *A;
    Value *B;
    Value *C;
};

enum class IndexKind {
    None,
    A_ik, // i*4 + k 
    B_kj, // k*4 + j
    C_ij  // i*4 + j
};

struct AccumulatorPattern {
    PHINode *AccPhi;
    BinaryOperator *FMul;
    BinaryOperator *FAdd;
};

class MlirToMatMulPass : public PassInfoMixin<MlirToMatMulPass> {
public:
    PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM) {
        errs() << "Running MlirToMatMulPass\n";
        bool Changed = false;
        auto &LI = AM.getResult<LoopAnalysis>(F);
        auto &DT = AM.getResult<DominatorTreeAnalysis>(F);
        auto &SE = AM.getResult<ScalarEvolutionAnalysis>(F);
/*         SmallVector<Loop*, 8> LoopList;
        LoopList = LI.getLoopsInPreorder();
        for (Loop *L : LoopList) {
            analyzeCanonicalLoop(L, SE);
        } */

        auto NestOpt = findMatmulLoopNest(LI, SE);
        if (!NestOpt) {
            errs() << "Did not find MatmulNest structure\n";
            return PreservedAnalyses::all();
        }
        const MatMulNest &Nest = *NestOpt;
        errs() << "Outer IV: "; Nest.Outer.IndVar->dump();
        errs() << "Middle IV: "; Nest.Middle.IndVar->dump();
        errs() << "Inner IV: "; Nest.Inner.IndVar->dump();
        PHINode *I = Nest.Outer.IndVar;
        PHINode *J = Nest.Middle.IndVar;
        PHINode *K = Nest.Inner.IndVar;

        auto &InnerLoop = Nest.Inner.L;
       // BasicBlock *Body = InnerLoop->getHeader()->getSingleSuccessor();
        BasicBlock *Body = InnerLoop->getLoopLatch();

        if (!Body) {
            return PreservedAnalyses::all();
        }

        SmallVector<Instruction*, 4> Candidates;
        for (Instruction &Inst : *Body) {
            auto *BO = dyn_cast<BinaryOperator>(&Inst);
            if (!BO) { continue; }
            if (BO->getOpcode() == Instruction::FMul) {
                LoadInst *LoadA = dyn_cast<LoadInst>(BO->getOperand(0));
                LoadInst *LoadB = dyn_cast<LoadInst>(BO->getOperand(1));
                if (!LoadA || !LoadB) { continue; }
                
                auto *GEPA = dyn_cast<GetElementPtrInst>(LoadA->getPointerOperand());
                auto *GEPB = dyn_cast<GetElementPtrInst>(LoadB->getPointerOperand());
                if (!GEPA || !GEPB) { continue; }

                // 1D GEP
                // TODO: Allow for other forms of GEP?
                if (GEPA->getNumIndices() != 1 || GEPB->getNumIndices() != 1) {
                    continue;
                }
                Value *IdxA = GEPA->getOperand(1);
                Value *IdxB = GEPB->getOperand(1);

                IndexKind KA = classifyIndex(IdxA, I, J, K);
                IndexKind KB = classifyIndex(IdxB, I, J, K);

                if (KA == IndexKind::A_ik && KB == IndexKind::B_kj) {
                    errs() << "FMul looks like A[i,k]*B[k,j]:\n";
                    BO->dump();
                }

                if (KA == IndexKind::B_kj && KB == IndexKind::A_ik) {
                    errs() << "FMul looks like B[k,j]*A[i,k] (swapped):\n";
                    BO->dump();
                }

                SmallVector<AccumulatorPattern, 4> APVec;
                findAccumulatorPatterns(Nest.Inner.L, APVec);
                if (APVec.empty()) { 
                    continue; 
                }
                std::optional<AccumulatorPattern> APOpt = matchAccumulatorPattern(APVec, BO);
                if (!APOpt) { 
                    continue;
                }
                AccumulatorPattern AP = *APOpt;
                errs() << "AP AccPhi:\n";
                AP.AccPhi->dump();
                errs() << "AP FMul:\n";
                AP.FMul->dump();
                errs() << "AP FAdd:\n";
                AP.FAdd->dump();
            }
            
        }

        return PreservedAnalyses::all();
    }

    std::optional<CanonicalLoop> analyzeCanonicalLoop(Loop *L, ScalarEvolution &SE) {
        assert(L && "Null Loop passed in");
        errs() << "Analyzing loop..."; L->dump();
        BasicBlock *preheader = L->getLoopPreheader();
        BasicBlock *header = L->getHeader();
        BasicBlock *latch = L->getLoopLatch();
        BasicBlock *exit = L->getExitingBlock();
        if (!preheader || !header || !latch || !exit) {
            return std::nullopt;
        }

        std::optional<PHINode*>indVarOpt = findIndunctionPhiInLoop(L);
        if (!indVarOpt) { return std::nullopt; }
        PHINode *indVar = *indVarOpt; // or indVarOpt.value(), not sure which is better
        //errs() << "Find indVar: ";
        //indVar->dump();
        const SCEV *S = SE.getSCEV(indVar);

        const auto *AR = dyn_cast<SCEVAddRecExpr>(S);
        if (!AR || AR->getLoop() != L) {
            // Not a simple affine IV for this loop
            return std::nullopt;
        }
        const SCEV *StartS = AR->getStart();
        const SCEV *StepS = AR->getStepRecurrence(SE);

        auto *StartC = dyn_cast<SCEVConstant>(StartS);
        auto *StepC = dyn_cast<SCEVConstant>(StepS);

        if (!StartC || !StepC) {
            return std::nullopt;
        }
        if (!StartC->getAPInt().isZero()) {
            return std::nullopt;
        }
        if (!StepC->getAPInt().isOne()) {
            return std::nullopt;
        }

        unsigned TripCount = SE.getSmallConstantTripCount(L);
        if (TripCount == 0) {
            return std::nullopt;
        }

        // Constant trip count, but we are manually getting the trip count
        // SCEV trip count is sometimes inaccurate
        auto *br = dyn_cast<BranchInst>(header->getTerminator());
        auto *cmp = dyn_cast<ICmpInst>(br->getCondition());
        auto *bound = dyn_cast<ConstantInt>(cmp->getOperand(1));
        if (!bound || bound->getZExtValue() != 4) {
            return std::nullopt;
        }

        return CanonicalLoop{L, indVar, bound->getZExtValue()};
    }

    // Find induction phi node
    // %iv = phi i64 [ 0, %preheader ], [ %iv.next, %latch ]
    // TODO: probably only need header
    std::optional<PHINode*> findIndunctionPhiInLoop(Loop *L) {
        BasicBlock *preheader = L->getLoopPreheader();
        BasicBlock *header = L->getHeader();
        BasicBlock *latch = L->getLoopLatch();
        BasicBlock *exit = L->getExitingBlock();
        if (!preheader || !header || !latch || !exit) {
            return std::nullopt;
        }

        for (PHINode &phi : header->phis()) {
            if (!phi.getType()->isIntegerTy()) {
                continue;
            }
            int preIdx = phi.getBasicBlockIndex(preheader);
            int latchIdx = phi.getBasicBlockIndex(latch);
            if (preIdx >= 0 && latchIdx >= 0) {
                return &phi;
            }
        }
        return std::nullopt;
    }

    // try match a single 3-deep perfect loop nest
    /*
    Outer  (i-loop)
    └─ Middle (j-loop)
        └─ Inner (k-loop)
    */
    std::optional<MatMulNest> findMatmulLoopNest(LoopInfo &LI, ScalarEvolution &SE) {
        // Find the top-level loop that is 3-deep nested
        for (Loop *outer : LI) {
            // top-level loop only
            if (outer->getParentLoop()) {
                continue;
            }
            if (outer->getSubLoops().size() != 1) {
                continue;
            }
            Loop *middle = outer->getSubLoops()[0];
            if (middle->getSubLoops().size() != 1) {
                continue;
            }
            Loop *inner = middle->getSubLoops()[0];
            if (inner->getSubLoops().size() != 0) {
                errs() << "subloop size is greater than\n";
                continue;
            }
            auto outerInfo = analyzeCanonicalLoop(outer, SE);
            auto middleInfo = analyzeCanonicalLoop(middle, SE);
            auto innerInfo = analyzeCanonicalLoop(inner, SE);
            if (!outerInfo) { errs() << "no outerInfo\n"; }
            if (!middleInfo) { errs() << "no middleInfo\n"; }
            if (!innerInfo) { errs() << "no innerInfo\n"; }

            if (!outerInfo || !middleInfo || !innerInfo) { continue; }
            
            // Check for trip count, requiring each to be 4
/*             if (!outerInfo->TripCount->equalsInt(4) 
                || !middleInfo->TripCount->equalsInt(4)
                || !outerInfo->TripCount->equalsInt(4)) {
                    continue;
            } */
            return MatMulNest{*outerInfo, *middleInfo, *innerInfo};
        }
        return std::nullopt;
    }

    void findAccumulatorPatterns(Loop *InnerLoop, SmallVector<AccumulatorPattern, 4>& Candidates);
    
    std::optional<AccumulatorPattern> matchAccumulatorPattern(SmallVector<AccumulatorPattern, 4> &Candidates, Value* FMul);

    IndexKind classifyIndex(Value *Idx,
                            PHINode *I,
                            PHINode *J,
                            PHINode *K);
};

std::optional<AccumulatorPattern> MlirToMatMulPass::matchAccumulatorPattern(SmallVector<AccumulatorPattern, 4> &Candidates,
                                                            Value *FMul) {

/*     // Get the value from Latch
    BasicBlock *Latch = InnerLoop->getLoopLatch();
    // At this point the Phi should have one value from Latch
    // Returning nullptr just in case it's called from unexpected places
    if (Phi->getBasicBlockIndex(Latch) < 0) {
        errs() << "The argument PhiNode should have an incoming value from loop latch\n";
        return std::nullopt;
    } */
    assert(FMul && "Null FMul as argument to matchAccumulatorPattern");
    BinaryOperator *FMulBOP = dyn_cast<BinaryOperator>(FMul);
    if (!FMulBOP || FMulBOP->getOpcode() != Instruction::FMul) {
        errs() << "The argument FMul should be an FMul instruction";
        return std::nullopt;
    }
            
    for (AccumulatorPattern &AP : Candidates) {
        BinaryOperator *FAdd = AP.FAdd;
        PHINode *AccumPhi = AP.AccPhi;
        if (FAdd->getOpcode() != Instruction::FAdd) {
            return std::nullopt;
        }
        Value *Op0 = FAdd->getOperand(0);
        Value *Op1 = FAdd->getOperand(1);
        if ((Op0 == AccumPhi && Op1 == FMul) 
            || (Op0 == FMul && Op1 == AccumPhi))  {
            AP.FMul = FMulBOP;
            return AP;
        }
    }      
    
    return std::nullopt;
                                                
}

/*
  %53 = phi float [ %67, %55 ], [ 0.000000e+00, %50 ]
  %67 = fadd float %53, %66
*/
void MlirToMatMulPass::findAccumulatorPatterns(Loop *InnerLoop, SmallVector<AccumulatorPattern, 4>& Candidates) {
    BasicBlock *Header = InnerLoop->getHeader();
    BasicBlock *Preheader = InnerLoop->getLoopPreheader();
    BasicBlock *Latch = InnerLoop->getLoopLatch();
    for (PHINode &Phi : Header->phis()) {
        if (!Phi.getType()->isFloatingPointTy()) {
            continue;
        }
        if (Phi.getNumIncomingValues() != 2) {
            continue;
        }
        // One value of this phi should come from the preheader
        int PreHeaderIdx = Phi.getBasicBlockIndex(Preheader);
        int LatchIdx = Phi.getBasicBlockIndex(Latch);
        if (PreHeaderIdx < 0 || LatchIdx < 0) {
            continue;
        }
        Value *PreHeaderVal = Phi.getIncomingValue(PreHeaderIdx);
        Value *LatchVal = Phi.getIncomingValue(LatchIdx);
        ConstantFP *CFP = dyn_cast<ConstantFP>(PreHeaderVal);
        if (!CFP || !(CFP->isExactlyValue(0.0f))) {
            continue;
        }
        Instruction *LatchInst = dyn_cast<Instruction>(LatchVal);
        if (!LatchInst || LatchInst->getParent() != Latch) {
            continue;
        }
        BinaryOperator *LatchBO = dyn_cast<BinaryOperator>(LatchInst);
        if (!LatchBO || LatchBO->getOpcode() != Instruction::FAdd) {
            continue;
        }

        Candidates.push_back(AccumulatorPattern{&Phi, nullptr, LatchBO});
        // Matched an FAdd in Latch at this point
        // Now check if the Latch has two operands;
        // One is the accumulator phi itself
        // The other is the fmul that we matched

    }
}


IndexKind MlirToMatMulPass::classifyIndex(Value *Idx,
                                          PHINode *I,
                                          PHINode *J,
                                          PHINode *K) {    
    // e.g. Idx: %58 = add i64 %57, %52
    /*
        %57 = mul i64 %44, 4 => 4*i 
        %52 = phi i64 [ %68, %55 ], [ 0, %50 ] => k
    */
    auto *Add = dyn_cast<BinaryOperator>(Idx);
    if (!Add || Add->getOpcode() != Instruction::Add) {
        return IndexKind::None;
    }

    Value *X = Add->getOperand(0);
    Value *Y = Add->getOperand(1);

    // V: %57 = mul i64 %44, 4 => 4*i
    // IV: %44 = phi i64 [ %76, %75 ], [ 0, %21 ]
    auto isMul4IV = [&](Value *V, Value *IV) -> bool {
        auto *Mul = dyn_cast<BinaryOperator>(V);
        if (!Mul || Mul->getOpcode() != Instruction::Mul) {
            return false;
        }

        Value *Op0 = Mul->getOperand(0);
        Value *Op1 = Mul->getOperand(1);

        ConstantInt *C = dyn_cast<ConstantInt>(Op0);
        Value *Other = Op1;
        if (!C) {
            C = dyn_cast<ConstantInt>(Op1);
            Other = Op0;
        }
        if (!C || !C->equalsInt(4)) {
            return false;
        }

        return Other == IV;
    };

    auto isIV = [&](Value *V, PHINode *IV) -> bool {
        return V == IV;
    };


    // 4i + k
    if (isMul4IV(X, I) && isIV(Y, K)) {
        return IndexKind::A_ik;
    }
    // k + 4i
    if (isMul4IV(Y, I) && isIV(X, K)) {
        return IndexKind::A_ik;
    }

    // 4k + j
    if (isMul4IV(X, K) && isIV(Y, J)) {
        return IndexKind::B_kj;
    }
    // j + 4k
    if (isMul4IV(Y, K) && isIV(X, J)) {
        return IndexKind::B_kj;
    }

    // 4i + j
    if (isMul4IV(X, I) && isIV(Y, J)) {
        return IndexKind::C_ij;
    }
    if (isMul4IV(Y, I) && isIV(X, J)) {
        return IndexKind::C_ij;
    }

    return IndexKind::None;
}

llvm::PassPluginLibraryInfo getMlirToMatMulPassPluginInfo() {
    return {LLVM_PLUGIN_API_VERSION, "MlirToMatMulPass",
            LLVM_VERSION_STRING,
            [] (PassBuilder &PB) {
                PB.registerPipelineParsingCallback(
                    [](StringRef Name, FunctionPassManager &FPM,
                        ArrayRef<PassBuilder::PipelineElement>) {
                            if (Name == "mlir2matmul") {
                                FPM.addPass(MlirToMatMulPass());
                                return true;
                            }
                            return false;
                    });
                // Ensure standard function analyses (Loop/DomTree/etc.) are registered
                PB.registerAnalysisRegistrationCallback(
                    [](FunctionAnalysisManager &FAM) {
                    //registerFunctionAnalyses(FAM);  // <-- critical line

                    FAM.registerPass([&] { return LoopAnalysis(); });
                  //  FAM.registerPass([&] { return DominatorTreeAnalysis(); });
                    FAM.registerPass([&] {return ScalarEvolutionAnalysis(); });
                  //  FAM.registerPass([&]{ return AssumptionAnalysis(); });
                    FAM.registerPass([&]{ return TargetIRAnalysis(); });
                    FAM.registerPass([&]{ return TargetLibraryAnalysis(); });
                });
            }};
}


extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
    return getMlirToMatMulPassPluginInfo();
}