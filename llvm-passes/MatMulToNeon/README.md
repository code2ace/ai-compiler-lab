# MatMulToNeon LLVM Pass

An LLVM function pass that transforms scalar 4×4 matrix multiplication kernels into vectorized NEON micro-kernels using ARM NEON SIMD instructions.

## Overview

This pass recognizes 4×4 matrix multiplication patterns in the innermost loop (k-loop) of nested loops and rewrites them to use:
- **Outer-product microkernel**: Re-expresses the k-loop as repeated rank-1 updates C += a(:,k) * b(k,:), enabling vector FMA updates to a register-resident C tile
- **Vector accumulators**: `<4 x float>` vector registers instead of 16 scalar accumulators
- **FMA intrinsics**: `llvm.fma.v4f32` (fused multiply-add) for efficient computation
- **Loop unrolling**: Fully unrolls the k-loop (4 iterations) to eliminate loop overhead
- **Vector shuffles**: Reorganizes matrix data for vector operations

The transformation replaces 16 scalar `fmul`+`fadd` reduction operations with 4 vectorized FMA calls per iteration, significantly improving performance on ARM/AArch64 architectures.

## Algorithm

### Pattern Recognition

The pass identifies:
1. **Innermost loops** (k-loop) that carry out matrix multiplication
2. **16 accumulator PHI nodes** representing matrix C elements (c00, c01, ..., c33)
3. **Reduction pattern**: Each accumulator follows `phi [0.0, entry], [fadd(phi, fmul(load_A, load_B)), latch]`
4. **Matrix indexing**: Infers row/column indices from GEP (GetElementPtr) operations

### Transformation Steps

1. **Collect Reduction PHIs**: Finds all float accumulator PHI nodes in the loop header
2. **Infer Matrix Layout**: Extracts row/column information from memory access patterns
3. **Build Vector Registers**:
   - Loads entire rows of matrix B into `<4 x float>` vectors
   - Reorganizes matrix A columns using shuffle operations
   - Creates 4 vector accumulators (one per column of C)
4. **Generate FMA Operations**: Replaces scalar operations with vector FMA calls:
   - For each row of B, splat each element into a vector register Vi
   - Multiply Vi with a column of A Vj
   - Accumulates Vi * Vj into C[:,K]
   ```llvm
   %acc0 = call fast <4 x float> @llvm.fma.v4f32(
       <4 x float> %A_col0,  // Column of A
       <4 x float> %B_row0_splat,  // Broadcasted row of B
       <4 x float> %acc0  // Accumulator
   )
   ```
5. **Loop Unrolling**: Fully unrolls the k-loop (4 iterations) and flattens the loop structure
6. **Replace Uses**: Extracts final results from vector accumulators and replaces original scalar PHI uses

## Requirements

- **LLVM 21+** (with new pass manager support)
- **Target Architecture**: ARM/AArch64 (for NEON support)
- **Matrix Size**: Fixed 4×4 matrices (`MAT_ROW=4`, `MAT_COL=4`)
- **Element Type**: `float` (32-bit floating point)
- **Function Signature**: Assumes matrix A is `arg[0]` and matrix B is `arg[1]`


## Building

The pass is built as an LLVM pass plugin:

```bash
cd llvm-passes
mkdir build && cd build
cmake -S ../ -G Ninja -DLLVM_DIR=/path/to/your/llvm/build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
ninja
```

This generates `libMatMulToNeon.dylib` (macOS) in the build directory.

## Usage

### Basic Usage

```bash
opt -load-pass-plugin /path/to/build/MatMulToNeon/MatMulToNeon.dylib -passes=matmul2neon -S input.ll -o output.ll
```

### Example Input and Workflow

The pass expects code structured like this (from `mat4x4_16acc.cpp`):

```cpp
void mat4x4_mul_16acc(const float* A, const float* B, float* C) {
    float c00=0.f, c01=0.f, ..., c33=0.f;
    
    for (int k = 0; k < 4; ++k) {
        float a0 = A[0*4 + k];
        float a1 = A[1*4 + k];
        // ... load a2, a3
        
        float b0 = B[k*4 + 0];
        // ... load b1, b2, b3
        
        // 16 scalar updates
        c00 += a0 * b0; c01 += a0 * b1; // ...
    }
    
    // Store results
    C[0*4 + 0] = c00; // ...
}
```
### Generate IR from Test File

```bash
clang++ -O0 -ffp-contract=off -S -emit-llvm mat4x4_16acc_test.cpp -o base.ll
```

### MemToReg and Canonicalization
```bash
opt -S -passes='mem2reg,instcombine,simplifycfg,loop-simplify,lcssa,indvars' base.ll -o dev.ll
```

### Run MatMulToNeon Pass
```bash
opt -load-pass-plugin /path/to/build/MatMulToNeon/MatMulToNeon.dylib -passes=matmul2neon -S dev.ll -o lowered.ll
```

### Light Clean-up to Help CodeGen
```bash
opt -S -passes='instcombine,simplifycfg' lowered.ll -o lowered_clean.ll
```

## Implementation Details

### Data Structures

- **`RCInfo`**: Represents a single accumulator update with row/column indices
  - `PHINode *PN`: The accumulator PHI node
  - `Instruction *Add`: The fadd instruction
  - `Instruction *Mul`: The fmul instruction
  - `Value *A`, `Value *B`: Loaded values from matrices A and B
  - `int Row`, `int Col`: Matrix position (0-3)

- **`LPInfo`**: Loop information
  - `Value *ABase`, `Value *BBase`: Base pointers to matrices
  - `PHINode *IVPhi`: Induction variable (k in the k-loop)

### Key Functions

- **`matchFaddOfFmulReduction`**: Pattern matches reduction PHI nodes
- **`assignRolColFor4x4`**: Infers matrix row/column indices from memory accesses
- **`tryInferRowColIndex`**: Extracts indices from GEP instructions
- **`tryRewrite4x4Kernel`**: Main transformation logic
- **`flattenFullyUnrolledLoop`**: Removes loop structure after unrolling
- **`replaceLCSSAWithClonedExtracts`**: Updates LCSSA PHI nodes after transformation

### Vector Shuffle Strategy

The pass uses shuffle operations to reorganize matrix A from row-major layout into column vectors:

```llvm
; Load 4 rows of A: [a00 a01 a02 a03], [a10 a11 a12 a13], ...
; Reorganize into 4 column vectors: [a00 a10 a20 a30], [a01 a11 a21 a31], ...
```

## Limitations & Known Issues

1. **Fixed Matrix Size**: Only handles 4×4 matrices
2. **Memory Layout Assumption**: Assumes row-major layout with specific GEP patterns
3. **Function Arguments**: Requires A as `arg[0]` and B as `arg[1]` (TODO: more robust handling)
4. **Single Block Loops**: Currently handles single-block loops only
5. **No Error Recovery**: Pattern matching failures cause the pass to exit without transformation
6. **LCSSA Handling**: Some edge cases in LCSSA PHI replacement may need refinement

## Testing

This project currently prioritizes end-to-end functional testing. Final output is compiled to an executable and validated by running the kernel.

Test files are located in `llvm-passes/test/CPP`:
- `dev.ll`: LLVM IR test input
- `lowered_clean.ll`: Sample output from MatMulToNeon pass
- `mat_test.cpp`: C++ source example, the result of the output kernel is compared against that of a reference implementation.

### Benchmark

Performance is measured in GFlops/Sec, in file `bench_mat4x4.cpp`, which first runs a small case to verify correctness. 

Run tests / benchmark:

### Link Output IR to an Executable
```bash
clang++ -O3 -ffast-math -mcpu=apple-m4 lowered_clean.ll  bench_mat4x4.cpp -o bench_pass
```

### Run Benchmark Specifying Number of Iterations
```bash
./bench_pass <# iterations default = 50000000>
```

### Sample Output From Benchmark
mat4x4_16acc_kernel: 99.07 GFLOP/s  (iters=50000000)

## References

- LLVM Vectorization Guide: https://llvm.org/docs/Vectorizers.html
- ARM NEON Intrinsics: https://developer.arm.com/architectures/instruction-sets/simd-isas/neon
- LLVM FMA Intrinsic: `llvm.fma.*` family of intrinsics

