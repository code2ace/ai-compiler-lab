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


                            errs() << "**************** START ******************" << "\n";
     //   errs() << APF << "\n";
     //   errs() << *C << "\n";
        //CI->dump();
                            errs() << "**************** END ******************" << "\n";

        //CI->print(errs()); errs() << "\n";


        /*
        for (Instruction *I : ToErase) {
            I->eraseFromParent();
        }
        */
        return PreservedAnalyses::none();
    }


    bool tryRewrite4x4Kernel(Function &F, Loop &L, ScalarEvolutionAnalysis::Result &SE, 
                            LoopInfo &LI, DominatorTree &DT, LPInfo &LP) {
        errs() << "Try Rewrite 4x4 Kernel\n";
        BasicBlock *Header = L.getHeader();
        BasicBlock *Body = Header; // deal with single-block loops for now
        BasicBlock *Latch = L.getLoopLatch();
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

        LLVMContext &Ctx = F.getContext();
        Type *F32 = Type::getFloatTy(Ctx);
        // Vector to hold a column of matrix A
        auto *V4F = FixedVectorType::get(F32, 4);

        // Create 4 vector phis
        SmallVector<PHINode*, 4> VecPNs;

        // Create 4 vector phis for odd iterations - 1, 3
        SmallVector<PHINode*, 4> VecPNsOdd;

        auto C1 = ConstantInt::get(IVPhi->getType(), 1);
        // Create IV+1 phi node, IV is k, so this is k+1
        // TODO: hasNUW and hasNSW set to true??
        auto *IVK1 = HB.CreateAdd(IVPhi, C1, IVPhi->getName()+Twine("k1"),
                                    false, false);

        // Populate rows of A (of one column) into one vector register outside of the loop below,
        // since it's updated by the induction variable
        SmallVector<RCInfo, 4> ColA = ColVecs[0];


        Value *ColAVec = PoisonValue::get(V4F);
        // ColK contains load instructions from column 0 of A, sorted by
        // row number
        for (int i = 0; i < ColA.size(); i++) {
            RCInfo &RC = ColA[i];
            // TODO: maybe change the index to Builder.getInt32(i)
            ColAVec = Builder.CreateInsertElement(ColAVec, RC.A, i);
        }

        /***** unrolling */
       

        // Unroll BVec creation
        // Same column, next row, indexed by indvars.iv + 1
        auto C0 = ConstantInt::get(IVK1->getType(), 0);
        auto C2 = ConstantInt::get(IVK1->getType(), 2);
        auto C4 = ConstantInt::get(IVK1->getType(), 4);
        auto C8 = ConstantInt::get(IVK1->getType(), 8);
        auto C12 = ConstantInt::get(IVK1->getType(), 12);


        // For building splats of B
        Value *B_k_idx = Builder.CreateShl(IVPhi, C2, "b_k_idx", true, true);
        Value *B_k1_idx = Builder.CreateAdd(B_k_idx, C4, "b_k1_idx", true, true);
        // Construct BBase for k and k+1
        Value *BBase_k = Builder.CreateInBoundsGEP(F32, LP.BBase, 
            B_k_idx, "BBase_k");
        Value *BBase_k1 = Builder.CreateInBoundsGEP(F32, LP.BBase, 
            B_k1_idx, "BBase_k1");


        // Construct GEP and load for B[IVK1*4+j]
        // Same column as indexed by indvars.iv, but the next row
        Value *IdxK1_4 = Builder.CreateMul(IVK1, C4, 
            "b_idx_k1x4",true, true);
        // Create vectors for storing the result phi nodes
        // So we can combine them later
        SmallVector<PHINode*, 4> CVecPhisK;
        SmallVector<PHINode*, 4> CVecPhisK_1;

        auto lane = [&](Value *Vec, int j) {
            return Builder.CreateExtractElement(Vec, Builder.getInt32(j));
        };

        auto splat = [&](Value *s) {
            return Builder.CreateVectorSplat(4, s);
        };

        auto *PtrV4Ty = PointerType::getUnqual(V4F->getContext());
        Value *Bv_ptr = Builder.CreateBitCast(BBase_k, PtrV4Ty);
        Value *Bv1_ptr = Builder.CreateBitCast(BBase_k1, PtrV4Ty);

        LoadInst *Bv = Builder.CreateAlignedLoad(V4F, Bv_ptr, Align(4),"brow.k");
        LoadInst *Bv1 = Builder.CreateAlignedLoad(V4F, Bv1_ptr, Align(4),"brow.k1");

        Value *bs_k[4], *bs_k1[4];
        for (int j = 0; j < 4; j++) {
            bs_k[j] = splat(lane(Bv, j));
            bs_k1[j] = splat(lane(Bv1, j));
        }


        // For building rows of A
        // Precompute row base for A
        Value *Arow0 = Builder.CreateInBoundsGEP(F32, LP.ABase, C0, "Arow0");
        Value *Arow1 = Builder.CreateInBoundsGEP(F32, LP.ABase, C4, "Arow1");
        Value *Arow2 = Builder.CreateInBoundsGEP(F32, LP.ABase, C8, "Arow2");
        Value *Arow3 = Builder.CreateInBoundsGEP(F32, LP.ABase, C12, "Arow3");
        
        Value *ARowBases[4] = {Arow0, Arow1, Arow2, Arow3};

        // Load each row of A into a vector (one vector at a time)
        // Store load instruction for each row of A
        Value *ARowLds[4];
        for (int i = 0; i < 4; i++) {
            Value *ARow_ptr = Builder.CreateBitCast(ARowBases[i], PtrV4Ty);
            ARowLds[i] = Builder.CreateAlignedLoad(V4F, ARow_ptr, 
                Align(4), "arow"+Twine(i));
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

        for (int r = 0; r < 4; r++) {
            // Unroll indvars.iv by 2, so create A row vec for next column of A (in the same loop body of the IR)
            //auto CI_row4 = ConstantInt::get(IVK1->getType(), r * 4);
            // Calculate index into A
            //Value *Idx = Builder.CreateAdd(CI_row4, IVK1, 
            //                        "a_idx_"+Twine(r)+ "_k1", true, true);
            // Get pointer value at A[Idx]
            /* Value *A_ptr_k1 = Builder.CreateGEP(F32, ARowBases[r], 
                            IVK1, "a_idx_"+Twine(r)+ "_k1_ptr");
             */
            // Create Load inst from the pointer value of A[Idx]
          /*   LoadInst *A_ptr_k1_ld = Builder.CreateLoad(F32, ARowLds[r], "a_"+Twine(r)+"_k1");
            A_ptr_k1_ld->setAlignment(Align(4));  */
            Value *ARowElmK = Builder.CreateExtractElement(ARowLds[r], 
                IVPhi, "ARow_"+Twine(r)+"_k");
            ColAVec_k = insertA(ColAVec_k, extractA(ARowLds[r], IVPhi), r);
            //ColAVec_k = Builder.CreateInsertElement(ColAVec_k, ARowElmK, r);
            ColAVec_k1 = insertA(ColAVec_k1, extractA(ARowLds[r], IVK1), r);
           // ColAVec_k1 = Builder.CreateInsertElement(ColAVec_k1, ARowLds[r], r);
        }
 

        for(int k = 0; k < 4; k++) {
            auto CVecK = PHINode::Create(V4F, 2,
            "cvec"+Twine(k), Header->getFirstNonPHIIt());
            SmallVector<RCInfo, 4> ColK = ColVecs[k];
            // CVecK_1 for unrolling K by 2
            auto CVecK_1 = PHINode::Create(V4F, 2,
            "cvec"+Twine(k)+"_1", Header->getFirstNonPHIIt());
            CVecPhisK.push_back(CVecK);
            CVecPhisK_1.push_back(CVecK_1);

            // Create vector phi
            auto *ZeroV = Constant::getNullValue(V4F);
            auto *Preheader = L.getLoopPreheader();

            CVecK->addIncoming(ZeroV, Preheader);
            CVecK_1->addIncoming(ZeroV, Preheader);

            // Used for splatting B
            // Using first phi in the ColK vector, since they
            // all share the same B column
           /*  RCInfo &RC = ColK[0];
            Value *ColSplat = Builder.CreateVectorSplat(4, RC.B);
            auto Cj = ConstantInt::get(IVK1->getType(), RC.Col);
 */
            /***** unrolling */
            // IVK1*4 + j
/*             Value *IdxB = Builder.CreateAdd(IdxK1_4, Cj, 
                "b_idx_k1x4j_"+Twine(k), true, true);
            
            // Get pointer address B[4(indvar.iv+1)+j] and load
            Value *B_ptr_k1 = Builder.CreateGEP(F32, LP.BBase, IdxB, "b_idx_k1_ptr");

            LoadInst *B_k1_ld = Builder.CreateAlignedLoad(F32, B_ptr_k1, 
                Align(4),"b_"+Twine(k)+"_j_k1");
             
            Value *B_ptr_k1 = Builder.CreateGEP(F32, BBase_k1, 
                ConstantInt::get(IVK1->getType(),k), "b_idx_k1_ptr");

            LoadInst *B_k1_ld = Builder.CreateAlignedLoad(F32, B_ptr_k1,
                Align(4));
            Value *ColSplat_k1 = Builder.CreateVectorSplat(4, B_k1_ld);
                */
            /***** unrolling ---*/
            

         /*    // Create load instruction for A[0][k+1]->A[3][k+1]
            // Create GEP for k+1 
            // GEP for k: %0 = getelementptr inbounds nuw float, ptr %A, i64 %indvars.iv
            Type *F32 = Type::getFloatTy(Builder.getContext());
            Value *RowA_K1 = Builder.CreateInBoundsGEP(F32, LP.ABase, IVK1, "A_k1");
 */
            // Preserve FMF - not sure if it's necessary 
            // TODO: double check 
            FastMathFlags FMF;
            if (auto *FPOp = dyn_cast<FPMathOperator>(ColK[0].Add)) {
                FMF = FPOp->getFastMathFlags();
                Builder.setFastMathFlags(FMF);
            }

            // Construct the FMA
            Function *FMA = Intrinsic::getOrInsertDeclaration(Header->getModule(),
                                Intrinsic::fma, {V4F});
            Value *ColAcc = Builder.CreateCall(FMA, {ColAVec, bs_k[k], CVecK});
            FMA->setAttributes(F.getAttributes());
            // ColAcc->dump();

            CVecK->addIncoming(ColAcc, Latch);
            VecPNs.push_back(CVecK);

            /***** unrolling */
            // TODO: probably should just reuse FMA from above?
            Function *FMA_k1 = Intrinsic::getOrInsertDeclaration(Header->getModule(), 
                                Intrinsic::fma, {V4F});
            Value *ColAcc_k1 = Builder.CreateCall(FMA_k1, {ColAVec_k1, bs_k1[k], CVecK_1});
            FMA_k1->setAttributes(F.getAttributes());
            CVecK_1->addIncoming(ColAcc_k1, Latch);
            // TODO: VecPN is probably no longer needed after unrolling
            VecPNs.push_back(CVecK_1);
            

        } // End for loop for one k (B[j][0] -> B[j][k])

        Value *CRes[4];
        // TODO: hardcoded 4
        for (int i = 0; i < 4; i++) {
            CRes[i] = HB.CreateFAdd(CVecPhisK[i], CVecPhisK_1[i]);
        }

        // Replace scalars with vector phi
        // Traverse through all phi nodes storing the result before 
        // vectorization
        // ColVecs groups phi nodes by column
        for (int k = 0; k < ColVecs.size(); k++) {
            auto &ColK = ColVecs[k];
            // Traverse each row in one column
            for (int i = 0; i < ColK.size(); i++) {
                auto &RC = ColK[i];
                auto *PN = RC.PN;

                //Value *ResVal = HB.CreateExtractElement(VecPNs[k], i);
                Value *ResVal = HB.CreateExtractElement(CRes[k], i);
     /*            errs() << "K " << k << "\n";
                errs() << "ResVal: ";
                ResVal->dump(); */
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
        Value *StepTwo = ConstantInt::get(IVPhi->getType(), 2);
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