#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Analysis/IVDescriptors.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/ExecutionEngine/GenericValue.h"
#include "llvm/IR/Analysis.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/FMF.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/IR/IntrinsicsAArch64.h"
#include "llvm/IR/ProfDataUtils.h"
#include "llvm/IR/Value.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Pass.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Passes/StandardInstrumentations.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/InstructionCost.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/IPO/HotColdSplitting.h"
#include "llvm/Transforms/Utils/LoopUtils.h"
#include "llvm/Transforms/Utils/Local.h"

#include <string>

// Deal with 4x4 matrices now
#define MAT_COL 4
#define MAT_ROW 4
#define ELEM_SIZE 4 // float size

using namespace llvm;
using namespace llvm::PatternMatch;

namespace {

// Row/col info of each target Phi Node update
struct RCInfo {
    PHINode *PN; // %c31.0 = phi float [ 0.000000e+00, %entry ], [ %add52, %for.inc ]
    Instruction *Add; // %add52 = fadd float %c31.0, %mul51
    Instruction *Mul; // %mul51 = fmul float %1, %10
    Value *A; // %1 (%1 = load float, ptr %arrayidx9, align 4)
    Value *B; // %10 (%10 = load float, ptr %arrayidx16, align 4)
    /*
    %0 = getelementptr inbounds nuw float, ptr %A, i64 %invdars.iv
    arrayidx9 = getelementptr inbounds nuw i8, ptr %0, i64 48
    => A[k+48] => A[3][k] (in matrix representation)
    */
    int Row = -1;

    /*
      %8 = shl nuw nsw i64 %indvars.iv, 2
      %9 = getelementptr inbounds nuw float, ptr %B, i64 %8
      %arrayidx16 = getelementptr inbounds nuw i8, ptr %9, i64 4
      %10 = load float, ptr %arrayidx16, align 4
    => B[4*k + 4] => B[k][1] (in matrix representation)
    */

    /*
    Example when col is 0
    %11 = shl nuw nsw i64 %indvars.iv, 2
    %arrayidx12 = getelementptr inbounds nuw float, ptr %B, i64 %11
    %12 = load float, ptr %arrayidx12, align 4
    */
    int Col = -1;
};

struct LPInfo {
    Value *ABase; // Base pointer to matrix %A
    Value *BBase; // Base pointer to matrix %B

    PHINode *IVPhi; // Induction Phi, for example k as in A[i][k] * B[k][j]
};

static int tryInferRowColIndex(Value *Ptr, LPInfo &LP) {
/*  For A: %1 = load float, ptr %arrayidx9, align 4
    A.getPointerOperand => %arrayidx9
    current parameter: %arrayidx9-> as GEP => get last constant => index
    index/12 == row index from A
 */
    if (auto *GEP = dyn_cast<GetElementPtrInst>(Ptr)) {
        Value *Index = GEP->getOperand(GEP->getNumOperands()-1);
        if (auto *ConstIntIdx = dyn_cast<ConstantInt>(Index)) {
            return static_cast<int>(ConstIntIdx->getZExtValue());
        } else if (match(GEP, m_GEP(m_Specific(LP.ABase), m_Specific(LP.IVPhi)))){
            return 0;
        /*
            Match for row 0 of Matrix A
            %arrayidx = getelementptr inbounds nuw float, ptr %A, i64 %indvars.iv
        */
          /*
            %11 = shl nuw nsw i64 %indvars.iv, 2
            %arrayidx12 = getelementptr inbounds nuw float, ptr %B, i64 %11
            column 0
          */ 
        } else if (Value *IVInst; match(GEP, m_GEP(m_Specific(LP.BBase), m_Value(IVInst)))) {
            // %11 = shl nuw nsw i64 %indvars.iv, 2 
            if (auto *BO = dyn_cast<BinaryOperator>(IVInst)) 
                if (BO->getOpcode() == Instruction::Shl && BO->getOperand(0) == LP.IVPhi) {
                    return 0;
                }
        }
    }

    return -1;
}

static bool assignRolColFor4x4(RCInfo &RC, LPInfo &LP) {
    // Grab the load instructions
    auto *AL = dyn_cast<LoadInst>(RC.A);
    auto *BL = dyn_cast<LoadInst>(RC.B);

    // TODO: handle cases where GEP is replaced with ExtractElement by
    // mem2ref/instcombine for AL
    
    // Row from %A
    // e.g., AL: %1 = load float, ptr %arrayidx9, align 4
    // pass %arrayidx9 to tryInferRowColIndex
    if (AL) {
        int idx = tryInferRowColIndex(AL->getPointerOperand(), LP);
        if (idx > -1) {
            RC.Row = idx / (MAT_COL * ELEM_SIZE);
            //errs() << "Row: " << RC.Row << "\n";
        } else {
            // Didn't successfully infer Row info
            // For now, print out message and exit
            errs() << "Did not successfully infer Row info for loadInst ";
            AL->dump();
            return false;
        }
    }

    if (BL) {
        int idx = tryInferRowColIndex(BL->getPointerOperand(), LP);
        if (idx > -1) {
            RC.Col = idx / ELEM_SIZE;
            //errs() << "Col: " << RC.Col << "\n";

        } else {
            // Same handling as Row
            errs() << "Did not successfully infer Col info for loadInst ";
            BL->dump();
            return false;
        }
    }
    return true;
}
// Check if this phi node is zero-initiated and of float value type
// E.g., %c00 = phi float [ 0.0, %entry ], [ %c00.n, %kloop ]
static bool isFloatAccPhi(PHINode *Phi) {
    if (!Phi || !Phi->getType()->isFloatTy()) {
        return false;
    }

    if (Phi->getNumIncomingValues() != 2) {
        return false;
    }

    // Make sure it's zero initiated
    if (auto *CF = dyn_cast<ConstantFP>(Phi->getIncomingValue(0))) {
        if (CF->isZero()) {
            errs() << "Is float acc phi\n";
            return true;
        }
    }
    return false;
}

static bool matchFaddOfFmulReduction(PHINode *PN, BasicBlock *Latch,
                                     RCInfo &RCResult) {
    // TODO: maybe use isFloatTy?
    if (!PN || !PN->getType()->isFloatingPointTy()) return false;

    // Find the latch update in the phi node
    // TODO: maybe need to check if the PN has exactly 2 incoming values
    int LatchIdx = PN->getBasicBlockIndex(Latch);

    if (LatchIdx < 0) return false;
    
    Value *LatchVal = PN->getIncomingValue(LatchIdx); 
    if (!match(LatchVal, m_FAdd(m_Value(), m_Value()))) return false;
    
    Instruction *AddI;
    AddI = dyn_cast<Instruction>(LatchVal);
    if (!AddI || AddI->getParent() != Latch) return false;
    errs() << "Add instruction: ";
    AddI->dump();

    Value *Prod = nullptr;
    if (!(match(LatchVal, m_FAdd(m_Specific(PN), m_Value(Prod))) ||
          match(LatchVal, m_FAdd(m_Value(Prod), m_Specific(PN))))) {
        return false;
    }

    // e.g., Prod ==   %mul51 = fmul float %1, %10
    Value *LHSop = nullptr, *RHSop = nullptr;
    if (!match(Prod, m_FMul(m_Value(LHSop), m_Value(RHSop)))) {
        return false;
    }
    Value *A, *B;
    Instruction *MulI;
    A = LHSop;
    B = RHSop;
    MulI = dyn_cast<Instruction>(Prod);
    RCResult = {PN, cast<Instruction>(LatchVal), cast<Instruction>(MulI), A, B, -1, -1};
    return true;
}

static void CleanUpScalarPhi(SmallVector<SmallVector<RCInfo, 4>, 4> &ColVecs) {
    for (int i = 0; i < ColVecs.size(); i++) {
        auto &ColVecI = ColVecs[i];
        for (int j = 0; j < ColVecI.size(); j++) {
            auto &RC = ColVecI[j];
            auto *PN = RC.PN;
            auto *AddI = RC.Add;
            auto *MulI = RC.Mul;
            if (PN->use_empty()) {
                errs() << "PN use empty" << "\n";
                PN->removeFromParent();
            }
         /*    if (AddI->use_empty()) {
                AddI->removeFromParent();
            }
            if (MulI->use_empty()) {
                MulI->removeFromParent();
            } */
        }
    }
}
// TODO: Remove, printing for debugging purpose
void printRC(const RCInfo &RC) {
    RC.A->dump();
    RC.B->dump();
    RC.Add->dump();
    RC.Mul->dump();
}

class MatMulToNeonPass : public PassInfoMixin<MatMulToNeonPass> {
public:
    PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM) {
        errs() << "Running MatMulToNeanPass" << "\n";
        bool Changed = false;
        Module *M = F.getParent();
        SmallVector<Instruction*, 8> ToErase;

        auto &LI = AM.getResult<LoopAnalysis>(F);
        auto &DT = AM.getResult<DominatorTreeAnalysis>(F);
       // auto &AC = FAM.getResult<llvm::AssumptionAnalysis>(F);     // returns AssumptionCache
        //auto &TLI = FAM.getResult<llvm::TargetLibraryAnalysis>(F); // returns TargetLibraryInfo
        auto &SE = AM.getResult<ScalarEvolutionAnalysis>(F);
        LPInfo LP;
        // Add function arguments, right now we assume 
        // that Matrix A is passed in as arg0 and B as arg1
        //assert(F.arg_size() >= 2 && "This function should have at least 2 arguments");
        if (F.arg_size() < 2) return PreservedAnalyses::all();
        // TODO: Not robust handling
        LP.ABase = F.getArg(0);
        LP.BBase = F.getArg(1);

        for (Loop *L : LI) {
            SmallVector<Loop*, 8> LoopList;
            //L->getInnerLoopsInPreorder(L, LoopList);
            LoopList = L->getLoopsInPreorder();
            for (Loop *subLoop : LoopList) {
                if (subLoop->getSubLoops().empty()) {
                    // found innermost loop
                    tryRewrite4x4Kernel(F, *subLoop, SE, LI, DT, LP);
                }
            }
        }
        return PreservedAnalyses::none();
    }


    bool tryRewrite4x4Kernel(Function &F, Loop &L, ScalarEvolutionAnalysis::Result &SE, 
                            LoopInfo &LI, DominatorTree &DT, LPInfo &LP) {
        errs() << "Try Rewrite 4x4 Kernel\n";
        BasicBlock *Header = L.getHeader();
        BasicBlock *Body = Header; // deal with single-block loops for now
        BasicBlock *Latch = L.getLoopLatch();
        BasicBlock *Exit = L.getExitBlock();
        BasicBlock *Preheader = L.getLoopPreheader();
        if (!Header) return false;

        // Collect candidate accumulator PHIs
        SmallVector<PHINode*, 16> Accs;

        RecurrenceDescriptor RD;
        InductionDescriptor ID;
        Value *A = nullptr, *B = nullptr;
        Instruction *AddI = nullptr, *MulI = nullptr;
        PHINode *IVPhi = nullptr;

        // Group target PHIs by cols
        SmallVector<RCInfo, 4> Col0Vec;
        SmallVector<RCInfo, 4> Col1Vec;
        SmallVector<RCInfo, 4> Col2Vec;
        SmallVector<RCInfo, 4> Col3Vec;

        SmallVector<SmallVector<RCInfo, 4>, 4> ColVecs(4);

        for (auto &I : *Header) {
            if (auto *Phi = dyn_cast<PHINode>(&I)) {
                errs() << "Checking phi nodes\n";
                RCInfo RC;
                if (InductionDescriptor::isInductionPHI(Phi, &L, &SE, ID)) {
                    LP.IVPhi = Phi;
                    IVPhi = Phi;
                }
                if (matchFaddOfFmulReduction(Phi, Latch, RC)) {
                    errs() << "Matched phi node\n";
                    //printRC(RC);
                    /*
                    At this point LP should contain info for ABase, BBase and Induction Phi
                    */
                    if(assignRolColFor4x4(RC, LP)) {
                        ColVecs[RC.Col].push_back(RC);
                        Accs.push_back(Phi);
                    } else {
                        return false;
                    }
                    //errs() << "RC.Col: " << RC.Col;

                }
            }
        }

        // TODO: move this check earlier
        if (Accs.size() < 16) { 
            return false; 
        }

        for (int i = 0; i < ColVecs.size(); i++) {
            sort(ColVecs[i], [](const RCInfo &rhs, const RCInfo &lhs) {
                return rhs.Row < lhs.Row;
            });
            errs() << "*********** Col: " << i << "**********\n";
            for (int j = 0; j < ColVecs[i].size(); j++) {
                printRC(ColVecs[i][j]);
                errs() << "Row: " << ColVecs[i][j].Row << "\n";
                errs() << "Col: " << ColVecs[i][j].Col << "\n";
            }
        }

        // IR builders, one for Latch, one for Header
        IRBuilder<> Builder(Latch->getTerminator());
        IRBuilder<> HB(&(*Header->getFirstNonPHIIt()));
        IRBuilder<> PB(&(*Preheader->getFirstInsertionPt()));

        LLVMContext &Ctx = F.getContext();
        Type *F32 = Type::getFloatTy(Ctx);
        // Vector to hold a column of matrix A
        auto *V4F = FixedVectorType::get(F32, 4);

        auto C1 = ConstantInt::get(IVPhi->getType(), 1);
        // Create IV+1 phi node, IV is k, so this is k+1
        // TODO: hasNUW and hasNSW set to true??
        auto *IVK1 = HB.CreateAdd(IVPhi, C1, IVPhi->getName()+Twine("k1"),
                                    false, false);

        // Populate rows of A (of one column) into one vector register outside of the loop below,
        // since it's updated by the induction variable
        SmallVector<RCInfo, 4> ColA = ColVecs[0];

        // Unroll BVec creation
        // Same column, next row, indexed by indvars.iv + 1
        auto C0 = ConstantInt::get(IVK1->getType(), 0);
        auto C2 = ConstantInt::get(IVK1->getType(), 2);
        auto C4 = ConstantInt::get(IVK1->getType(), 4);
        auto C8 = ConstantInt::get(IVK1->getType(), 8);
        auto C12 = ConstantInt::get(IVK1->getType(), 12);


        // Build BBases
        Value *BRowBases[4];
        
        BRowBases[0] = Builder.CreateInBoundsGEP(F32, LP.BBase, C0, "BRow0");
        BRowBases[1] = Builder.CreateInBoundsGEP(F32, LP.BBase, C4, "BRow1");
        BRowBases[2] = Builder.CreateInBoundsGEP(F32, LP.BBase, C8, "BRow2");
        BRowBases[3] = Builder.CreateInBoundsGEP(F32, LP.BBase, C12, "BRow3");

        SmallVector<PHINode*, 4> CVecPhisK;
        //SmallVector<PHINode*, 4> CVecPhisK_1;

        auto *PtrV4Ty = PointerType::getUnqual(V4F->getContext());

        // Load instructions for each row of B
        Value *BRowLds[4];

        for (int i = 0; i < 4; i++) {
            Value *BRow_ptr = Builder.CreateBitCast(BRowBases[i], PtrV4Ty);
            BRowLds[i] = Builder.CreateAlignedLoad(V4F, BRow_ptr, 
                Align(16), "brow"+Twine(i));
        }
/* 
        Value *bs_k[4], *bs_k1[4];
        for (int j = 0; j < 4; j++) {
            bs_k[j] = splat(lane(Bv, j));
            bs_k1[j] = splat(lane(Bv1, j));
        } */


        // For building rows of A
        // Precompute row base for A
        Value *Arow0 = PB.CreateInBoundsGEP(F32, LP.ABase, C0, "Arow0");
        Value *Arow1 = PB.CreateInBoundsGEP(F32, LP.ABase, C4, "Arow1");
        Value *Arow2 = PB.CreateInBoundsGEP(F32, LP.ABase, C8, "Arow2");
        Value *Arow3 = PB.CreateInBoundsGEP(F32, LP.ABase, C12, "Arow3");
        
        Value *ARowBases[4] = {Arow0, Arow1, Arow2, Arow3};

        // Load each row of A into a vector (one vector at a time)
        // Store load instruction for each row of A
        Value *ARowLds[4];
        for (int i = 0; i < 4; i++) {
            Value *ARow_ptr = PB.CreateBitCast(ARowBases[i], PtrV4Ty);
            ARowLds[i] = PB.CreateAlignedLoad(V4F, ARow_ptr, 
                Align(16), "arow"+Twine(i));
        }

        // TODO: hard coded 4 since we are doing 4x4 tiling
        Value *ColAVec_k = PoisonValue::get(V4F);
        Value *ColAVec_k1 = PoisonValue::get(V4F);

        auto extractA = [&](Value *Vec, Value *Idx) {
            return Builder.CreateExtractElement(Vec, Idx);
        };

        auto insertA = [&](Value *Vec, Value *Elm, int Idx) {
            return Builder.CreateInsertElement(Vec, Elm, Idx);
        };

        auto SV = [&](ArrayRef<int> Mask, Value *A, Value *B) {
            SmallVector<int, 8> M(Mask.begin(), Mask.end());
            return Builder.CreateShuffleVector(A, B, M);
        };

        Value *A_r0 = ARowLds[0];
        Value *A_r1 = ARowLds[1];
        Value *A_r2 = ARowLds[2];
        Value *A_r3 = ARowLds[3];

        Value *t0 = SV({0,4,1,5}, A_r0, A_r1);
        Value *t1 = SV({0,4,1,5}, A_r2, A_r3);
        Value *t2 = SV({2,6,3,7}, A_r0, A_r1);
        Value *t3 = SV({2,6,3,7}, A_r2, A_r3);

        Value *col0 = SV({0,1,4,5}, t0, t1); // [a00, a10, a20, a30]
        Value *col1 = SV({2,3,6,7}, t0, t1); // [a01, a11, a21, a31]
        Value *col2 = SV({0,1,4,5}, t2, t3); // [a02, a12, a22, a32]
        Value *col3 = SV({2,3,6,7}, t2, t3); // [a03, a13, a23, a33]
        
        Value *ColAVecs[4];
        ColAVecs[0] = col0;
        ColAVecs[1] = col1;
        ColAVecs[2] = col2;
        ColAVecs[3] = col3;

        for(int k = 0; k < 4; k++) {
            auto CVecK = PHINode::Create(V4F, 2,
            "cvec"+Twine(k), Header->getFirstNonPHIIt());
            SmallVector<RCInfo, 4> ColK = ColVecs[k];
            // CVecK_1 for unrolling K by 2
/*             auto CVecK_1 = PHINode::Create(V4F, 2,
            "cvec"+Twine(k)+"_1", Header->getFirstNonPHIIt()); */
            CVecPhisK.push_back(CVecK);
          //  CVecPhisK_1.push_back(CVecK_1);

            // Create vector phi
            auto *ZeroV = Constant::getNullValue(V4F);
            auto *Preheader = L.getLoopPreheader();

            CVecK->addIncoming(ZeroV, Preheader);       

        } // End for loop for one k (B[j][0] -> B[j][k])

        //Value *CVecK_vec[4];
        //Value *CVecK1_vec[4];
        
        auto broadcastLane = [&] (Value *brow, int idx, 
            IRBuilder<>&Builder) {
            assert(idx < 4 && "Lane index must be 0,1,2,3");
            SmallVector<int, 4> Mask = {idx, idx, idx, idx};
            return Builder.CreateShuffleVector(brow, brow, Mask);
        };

        auto lane = [&](Value *Vec, int j) {
            return Builder.CreateExtractElement(Vec, Builder.getInt32(j));
        };

        auto splat = [&](Value *s) {
            return Builder.CreateVectorSplat(4, s);
        };

        
        FastMathFlags FMF; 
        FMF.setFast();
        Builder.setFastMathFlags(FMF);

        Function *FMA = Intrinsic::getOrInsertDeclaration(Header->getModule(),
                Intrinsic::fma, {V4F});

        Value *Acc[4];
        for (int i = 0; i < 4; i++) {
            Acc[i] = CVecPhisK[i];
        }

        SmallVector<SmallVector<Value*, 4>, 4> BRowSplats;
        for (int i = 0; i < 4; i++) {
            //Value *BRow_i = BRowLds[i];
            // Row splat for row i of B
            // 4 vectors splats each index (col) of row i
            SmallVector<Value*, 4> BRowSplat_i;
            Value *curr;
            //curr = splat(lane(BRowLds[i], 0));
            curr = broadcastLane(BRowLds[i], 0, Builder);
            BRowSplat_i.push_back(curr);

            //curr = splat(lane(BRowLds[i], 1));
            curr = broadcastLane(BRowLds[i], 1, Builder);
            BRowSplat_i.push_back(curr);

            //curr = splat(lane(BRowLds[i], 2));
            curr = broadcastLane(BRowLds[i], 2, Builder);
            BRowSplat_i.push_back(curr);

            //curr = splat(lane(BRowLds[i], 3));
            curr = broadcastLane(BRowLds[i], 3, Builder);
            BRowSplat_i.push_back(curr);

            BRowSplats.push_back(BRowSplat_i);
        }

        for (int i = 0; i < 4; i++) {
            // V0: fma of column of A and first element of row B splatted
      /*       Value *BRowSplat0 = splat(lane(BRowLds[i], 0));
            Value *BRowSplat1 = splat(lane(BRowLds[i], 1));
            Value *BRowSplat2 = splat(lane(BRowLds[i], 2));
            Value *BRowSplat3 = splat(lane(BRowLds[i], 3));
 */            
            CallInst *V0 = Builder.CreateCall(FMA, {ColAVecs[i], 
                BRowSplats[i][0], Acc[0]});
            V0->setFastMathFlags(FMF);
            Acc[0] = V0;

            CallInst *V1 = Builder.CreateCall(FMA, {ColAVecs[i], 
                BRowSplats[i][1], Acc[1]});
            V1->setFastMathFlags(FMF);
            Acc[1] = V1;
            
            CallInst *V2 = Builder.CreateCall(FMA, {ColAVecs[i], 
                BRowSplats[i][2], Acc[2]});
            V2->setFastMathFlags(FMF);
            Acc[2] = V2;
            
            CallInst *V3 = Builder.CreateCall(FMA, {ColAVecs[i], 
                BRowSplats[i][3], Acc[3]});
            V3->setFastMathFlags(FMF);
            Acc[3] = V3;
            
            //VecPNs.push_back(CVecK_1);

            //CVecK_vec[i]
        }

        for (int i = 0; i < 4; i++) {
            CVecPhisK[i]->addIncoming(Acc[i], Latch);
        }

        // TODO: Move extraelement of final results out of the header
        // and into the loop exit block
        // Replace scalars with vector phi
        // Traverse through all phi nodes storing the result before 
        // vectorization
        IRBuilder<> EB(&(*Exit->getFirstNonPHIIt())); 

        Value *CRes[4];
        // TODO: hardcoded 4
/*         for (int i = 0; i < 4; i++) {
            CRes[i] = HB.CreateFAdd(CVecPhisK[i], CVecPhisK_1[i]);
        } */

        // ColVecs groups phi nodes by column
        for (int k = 0; k < ColVecs.size(); k++) {
            auto &ColK = ColVecs[k];
            // Traverse each row in one column
            for (int i = 0; i < ColK.size(); i++) {
                auto &RC = ColK[i];
                auto *PN = RC.PN;

                //Value *ResVal = HB.CreateExtractElement(VecPNs[k], i);
                //Value *ResVal = HB.CreateExtractElement(CRes[k], i);
                Value *ResVal = HB.CreateExtractElement(CVecPhisK[k], i);
                // Replace all uses inside the loop
                PN->replaceAllUsesWith(ResVal);
                bool Changed = formLCSSA(L, DT, &LI, &SE);
                if (Changed) {
                    errs() << "Loop changed by LCSSA" << "\n";
                }

               RecursivelyDeleteTriviallyDeadInstructions(PN);

            }
        } // End of for loop

        // For unrolling, change IV increment to +2
        // TODO: remove ind.var for full unrolling
        Value *StepTwo = ConstantInt::get(IVPhi->getType(), 4);
        Value *NewPhiUpdate = Builder.CreateAdd(IVPhi, StepTwo, "iv.next2", "true", "true");
        IVPhi->setIncomingValueForBlock(Latch, NewPhiUpdate);

        /******* Unrolling */
        // Combine the results from k and k+1 


       // CleanUpScalarPhi(ColVecs);

        return true;
    }
};

} // end of anonymous namespace

llvm::PassPluginLibraryInfo getMatMulToNeonPassPluginInfo() {
    return {LLVM_PLUGIN_API_VERSION, "MatMulToNeonPass",
            LLVM_VERSION_STRING,
            [] (PassBuilder &PB) {
                PB.registerPipelineParsingCallback(
                    [](StringRef Name, FunctionPassManager &FPM,
                        ArrayRef<PassBuilder::PipelineElement>) {
                            if (Name == "matmul2neon") {
                                FPM.addPass(MatMulToNeonPass());
                                return true;
                            }
                            return false;
                    });
                // Ensure standard function analyses (Loop/DomTree/etc.) are registered
                PB.registerAnalysisRegistrationCallback(
                    [](FunctionAnalysisManager &FAM) {
                    //registerFunctionAnalyses(FAM);  // <-- critical line

                    FAM.registerPass([&] { return LoopAnalysis(); });
                    FAM.registerPass([&] { return DominatorTreeAnalysis(); });
                    FAM.registerPass([&] {return ScalarEvolutionAnalysis(); });
                    FAM.registerPass([&]{ return AssumptionAnalysis(); });
                    FAM.registerPass([&]{ return TargetIRAnalysis(); });
                    FAM.registerPass([&]{ return TargetLibraryAnalysis(); });
                });
            }};
}

extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
    return getMatMulToNeonPassPluginInfo();
}