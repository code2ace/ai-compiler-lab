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
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
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
};

struct MatMulNest {
    CanonicalLoop Outer;
    CanonicalLoop Middle;
    CanonicalLoop Inner;
};

struct MatMulBase {
    Value *A = nullptr;
    Value *B = nullptr;
    Value *C = nullptr;
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

        auto NestOpt = findMatmulLoopNest(LI, SE);
        if (!NestOpt) {
            errs() << "Did not find MatmulNest structure\n";
            return PreservedAnalyses::all();
        }
        const MatMulNest &Nest = *NestOpt;

        PHINode *I = Nest.Outer.IndVar;
        PHINode *J = Nest.Middle.IndVar;
        PHINode *K = Nest.Inner.IndVar;

        Loop *InnerLoop = Nest.Inner.L;
        Loop *MiddleLoop = Nest.Middle.L;
        Loop *OuterLoop = Nest.Outer.L;
        BasicBlock *Body = InnerLoop->getLoopLatch();

        if (!Body) {
            return PreservedAnalyses::all();
        }

        SmallVector<Instruction*, 4> Candidates;
        AccumulatorPattern AP;
        MatMulBase MBase;
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
                // Get ABase and BBase from GEPA and GEPB
                MBase.A = GEPA->getPointerOperand();
                MBase.B = GEPB->getPointerOperand();

                SmallVector<AccumulatorPattern, 4> APVec;
                findAccumulatorPatterns(Nest.Inner.L, APVec);
                if (APVec.empty()) { 
                    continue; 
                }
                std::optional<AccumulatorPattern> APOpt = matchAccumulatorPattern(APVec, BO);
                if (!APOpt) { 
                    continue;
                }
                AP = *APOpt;
/*                 errs() << "AP AccPhi:\n";
                AP.AccPhi->dump();
                errs() << "AP FMul:\n";
                AP.FMul->dump();
                errs() << "AP FAdd:\n";
                AP.FAdd->dump(); */

                std::optional<std::pair<Value*, StoreInst*>> COpt 
                    = matchCStore(InnerLoop, AP.AccPhi, I, J, K);
                if (!COpt) { continue; }
                Value *CBase = COpt->first;
                errs() << "CBase: "; CBase->dump();
                MBase.C = CBase;
            }
            
        }
        Module *M = F.getParent();
        std::optional<Function*>KernelOpt = createKernelCall(M, OuterLoop, MBase);
        if (!KernelOpt) {
            return PreservedAnalyses::all();
        }
        Function *Kernel = *KernelOpt;

        Value *ABase = MBase.A;
        Value *BBase = MBase.B;
        Value *CBase = MBase.C;
        BasicBlock *OuterPH = OuterLoop->getLoopPreheader();
        assert(OuterPH && "Outer loop must be canonical and have a preheader\n");

        // Hoist ABase, BBase and CBase into the preheader
        if (!hoistValueToPreheader(ABase, OuterLoop, OuterPH)) {
            errs() << "Unable to hoist ABase value: "; ABase->dump();
            return PreservedAnalyses::none();
        }
        if (!hoistValueToPreheader(BBase, OuterLoop, OuterPH)) {
            errs() << "Unable to hoist BBase value: "; BBase->dump();
            return PreservedAnalyses::none();
        }
        if (!hoistValueToPreheader(CBase, OuterLoop, OuterPH)) {
            errs() << "Unable to hoist CBase value: "; CBase->dump();
            return PreservedAnalyses::none();
        }

        IRBuilder<> B(OuterPH->getTerminator());
        SmallVector<Value*, 3> Args = {ABase, BBase, CBase };
        B.CreateCall(Kernel, Args);
        
        return PreservedAnalyses::all();
    }

    std::optional<CanonicalLoop> analyzeCanonicalLoop(Loop *L, ScalarEvolution &SE) {
        assert(L && "Null Loop passed in");
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
                continue;
            }
            auto outerInfo = analyzeCanonicalLoop(outer, SE);
            auto middleInfo = analyzeCanonicalLoop(middle, SE);
            auto innerInfo = analyzeCanonicalLoop(inner, SE);
            if (!outerInfo) { errs() << "no outerInfo\n"; }
            if (!middleInfo) { errs() << "no middleInfo\n"; }
            if (!innerInfo) { errs() << "no innerInfo\n"; }

            if (!outerInfo || !middleInfo || !innerInfo) { continue; }
            
            return MatMulNest{*outerInfo, *middleInfo, *innerInfo};
        }
        return std::nullopt;
    }

    std::optional<Function*> createKernelCall(Module *M, Loop *OuterLoop, MatMulBase &MBase);

    // Returns true if a move is successful
    // Returns false if V cannot be safely hoisted or is non-instruction
    bool hoistValueToPreheader(Value *V, Loop *OuterLoop, BasicBlock *OuterPreheader,
                               DominatorTree *DT = nullptr,
                               SmallPtrSetImpl<Instruction*> *Visited = nullptr);

    void findAccumulatorPatterns(Loop *InnerLoop, SmallVector<AccumulatorPattern, 4>& Candidates);
    
    std::optional<AccumulatorPattern> matchAccumulatorPattern(SmallVector<AccumulatorPattern, 4> &Candidates, Value* FMul);
    std::optional<std::pair<Value*, StoreInst*>> matchCStore(Loop *InnerLoop, PHINode *AccPhi,
                                                             PHINode *I, PHINode *J, PHINode *K);

    IndexKind classifyIndex(Value *Idx,
                            PHINode *I,
                            PHINode *J,
                            PHINode *K);
};

std::optional<Function*> MlirToMatMulPass::createKernelCall(Module *M, Loop *OuterLoop, MatMulBase &MBase) {
    BasicBlock *Preheader = OuterLoop->getLoopPreheader();
    if (!Preheader) {
        errs() << "Outer loop has no preheader!";
        return std::nullopt;
    }

    Value *ABase = MBase.A;
    Value *BBase = MBase.B;
    Value *CBase = MBase.C;
    assert(ABase && BBase && CBase && "Nullptr in base pointers of A, B and C\n");

    Function *Kernel = M->getFunction("mat4x4_16acc_kernel");
    // declare: void(@AType*, @BType*, @CType*)
    if (!Kernel) {
        // Apparently A, B and C should all be float ptr types
        // Still dynamically setting the types here
        // Relying on callers of this function for type checking
        FunctionType *FT = FunctionType::get(Type::getVoidTy(M->getContext()),
                                            { ABase->getType(), BBase->getType(), 
                                              CBase->getType()}, false);
        Kernel = Function::Create(FT, Function::ExternalLinkage, "mat4x4_16acc_kernel", M);
    }
    return Kernel;
}

bool MlirToMatMulPass::hoistValueToPreheader(Value *V, Loop *OuterLoop, BasicBlock *OuterPreheader,
                                             DominatorTree *DT,
                                             SmallPtrSetImpl<Instruction*> *Visited) {
    Instruction *I = dyn_cast<Instruction>(V);
    if (!I) return true; // constants/args/globals already outside

    if (!OuterLoop->contains(I)) return true;

    // Cannot hoist out of the loop if it isn't loop-invariant
    //if (!OuterLoop->isLoopInvariant(I)) return false;

    SmallPtrSet<Instruction*, 8> LocalVisited;
    if (!Visited) {
        Visited = &LocalVisited;
    }

    // Break out of cycles
    if (!Visited->insert(I).second) {
        return false;
    }

    // TODO: maybe relax from mayHaveSideEffects a bit
    // to allow pure bitcasts, ptrtoint, etc.
    // For now, just let mayHaveSideEffects decide
    if (I->mayHaveSideEffects() || isa<CallInst>(I) || isa<StoreInst>(I)) {
        return false;
    }

    if (auto *LI = dyn_cast<LoadInst>(I)) {
        if (LI->isAtomic() || LI->isVolatile()) {
            return false;
        }
    }
    // Hoist instructions like this:
    //  %70 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %28, 1
    // Hoist operand first, for example %28 here
    for (Use &U : I->operands()) {
        Value *Op = U.get();
        Instruction *OpI = dyn_cast<Instruction>(Op);
        if (!OpI) { continue; } // constants, args are fine, skipping
        if (OpI->getParent() == OuterPreheader) {
            continue;
        }
        // recursively hoist operand if it's inside the loop
        if (OuterLoop->contains(OpI)) {
            if (!hoistValueToPreheader(OpI, OuterLoop, OuterPreheader, DT, Visited)) {
                return false;
            }
            // TODO: Being defensive here but I don't know if it's necessary
            if (OuterLoop->contains(OpI)) {
                return false;
            }
        }
    }

    // Now move I before the preheader terminator
    if (I->getParent() != OuterPreheader) {
        I->moveBefore(OuterPreheader->getTerminator()->getIterator());
    }

    // Dominance check
    if (DT) {
        for (User *U : I->users()) {
            if (Instruction *UI = dyn_cast<Instruction>(U)) {
                if (!DT->dominates(I->getParent(), UI->getParent())) {
                    return false;
                }
            }
        }
    }

    return true;
}

std::optional<AccumulatorPattern> MlirToMatMulPass::matchAccumulatorPattern(SmallVector<AccumulatorPattern, 4> &Candidates,
                                                            Value *FMul) {

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

std::optional<std::pair<Value*, StoreInst*>> MlirToMatMulPass::matchCStore(Loop *InnerLoop, 
                                                         PHINode *AccPhi,
                                                         PHINode *I, 
                                                         PHINode *J, 
                                                         PHINode *K) {
    // Get the exit block
    // The store instruction should be in the exit block (assumption?)
    BasicBlock *Exit = InnerLoop->getExitBlock();
    if (!Exit) {
        errs() << "No exit block found for inner loop\n";
        return std::nullopt;
    }

    for (Instruction &Inst : *Exit) {
        auto *St = dyn_cast<StoreInst>(&Inst);
        if (!St) {
            continue;
        }

        // The value being stored should be the accumulator phi
        Value *StVal = St->getValueOperand();
        if (StVal != AccPhi) {
            continue;
        }

        Value *StPtr = St->getPointerOperand();
        auto *GEP = dyn_cast<GetElementPtrInst>(StPtr);
        if (!GEP) { continue; }
        if (GEP->getNumIndices() != 1) { continue; }

        // Get the base and index of C
        Value *Idx = GEP->getOperand(1);
        IndexKind KC = classifyIndex(Idx, I, J, K);
        if (KC != IndexKind::C_ij) { continue; }
        // TODO: verify that the base is the CBase from MatMulBase
        Value *CBase = GEP->getPointerOperand();
        return std::make_pair(CBase, St);
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