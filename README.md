# AI Compiler Lab

A small collection of **LLVM passes** and **MLIR experiments** for optimizing **4×4 FP32 matrix multiplication micro-kernels** (GEMM-style), with a focus on **AArch64 / ARM NEON** codegen.

This root README is intentionally brief (landing page). Detailed design notes live in the component READMEs.

## Highlights

- LLVM IR pattern recognition + rewriting (SSA / DominatorTree / LoopInfo-aware)
- SIMD shaping for NEON (vector FMAs, register-tiled micro-kernel form)
- MLIR dialect + lowering pipeline experiments for fixed 4×4 kernels

## Components

### MatMulToNeon (LLVM pass)
- **Path**: `llvm-passes/MatMulToNeon/`
- **Docs**: `llvm-passes/MatMulToNeon/README.md`

Recognizes a 4×4 micro-kernel shape (16 scalar accumulator updates in an inner `k` loop) and rewrites it into a SIMD-friendly form using `<4 x float>` vector operations / FMA-style accumulation. Includes an optional full unroll / loop-flatten experiment while preserving SSA + CFG invariants.

See [llvm-passes/MatMulToNeon/README.md](llvm-passes/MatMulToNeon/README.md) for detailed documentation.

### MlirToMatMul (LLVM pass)

Detects canonical `i/j/k` loop nests implementing matmul and replaces them with calls to micro-kernel functions.

### qu4d-mlir (custom MLIR dialect)
- **Path**: `qu4d-mlir/`
- **Docs**: `qu4d-mlir/README.md`

**Location**: `qu4d-mlir/`

A small MLIR dialect + passes for expressing and lowering fixed-shape 4×4 kernels.

See [qu4d-mlir/README.md](qu4d-mlir/README.md) for detailed documentation.

## Repo layout
```text
ai-compiler-lab/
├── llvm-passes/
│   ├── MatMulToNeon/
│   ├── MlirToMatMul/
│   ├── StrengthReduceLoop/
│   └── test/
└── qu4d-mlir/
```

## References

- [LLVM Vectorization Guide](https://llvm.org/docs/Vectorizers.html)
- [MLIR Documentation](https://mlir.llvm.org/)
- [ARM NEON Intrinsics](https://developer.arm.com/architectures/instruction-sets/simd-isas/neon)
- [LLVM Pass Writing Guide](https://llvm.org/docs/WritingAnLLVMNewPMPass.html)
- [MLIR Dialect Development](https://mlir.llvm.org/docs/Tutorials/DefiningDialects/Operations/)
