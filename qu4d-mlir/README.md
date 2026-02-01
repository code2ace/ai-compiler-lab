# Qu4d-MLIR Custom Dialect

A custom MLIR dialect and transformation pipeline for optimizing 4×4 matrix multiplication kernels.

## Overview

Qu4d is a small custom MLIR dialect designed to represent and optimize fixed-size 4×4 matrix multiplication operations. The goal is to make 4x4 kernels explicit at a higher level than LLVM IR, enabling targeted transforms and cleaner lowering pipelines.

## What's Inside

- **`qu4d` dialect**
  - `qu4d.matmul_4x4`: fixed 4×4 microkernel op (shape/type verified)
- **Passes**
  - `qu4d-from-linalg`: match `linalg.matmul` (4×4×f32) → `qu4d.matmul_4x4`
  - `qu4d-lower-matmul4x4-scf`: lower `qu4d.matmul_4x4` → canonical `scf.for` i/j/k loops
- **Tooling**
  - `qu4d-opt`: an `opt`-like driver for running the dialect + passes

### Transformation Pipeline

```
Linalg Dialect → qu4d Dialect → SCF Dialect → Arith Dialect → LLVM
```

1. **High-Level Matching** (`qu4d-from-linalg`)
   - Pattern matches `linalg.matmul` operations
   - Verifies 4×4×f32 memref shapes
   - Converts to `qu4d.matmul_4x4` operations

2. **Lowering** (`lower-matmul-4x4`)
   - Lowers `qu4d.matmul_4x4` to nested SCF loops
   - Generates canonical 3-nested loop structure (i, j, k)
   - Produces memref load/store operations

3. **Micro Kernel Delegation** (`mlir2matmul`)
   - Recognizes 3-nested loop structures implementing matrix multiplication
   - Matches and classifies accumulator and memory access patterns
   - Replaces the matched matrix multiplication with calls to optimized micro-kernels

## Building

### Prerequisites

- **MLIR** (from LLVM project)
- **LLVM** (required by MLIR)
- **CMake 3.20+**
- **C++17** compatible compiler
- **Ninja**

### Build Instructions

```bash
cd qu4d-mlir
mkdir build && cd build

# Configure with MLIR
cmake -G Ninja .. \  
-DMLIR_DIR=/path/to/mlir/lib/cmake/mlir \  
-DLLVM_DIR=/path/to/llvm/lib/cmake/llvm \  
-DMLIR_INCLUDE_DIR=/path/to/llvm/mlir/include \  
-DMLIR_BUILD_INCLUDE_DIR=/path/to/llvm/build/tools/mlir/include \  
-DCMAKE_EXPORT_COMPILE_COMMANDS=ON

# Build the qu4d-opt tool and Qu4d dialect
ninja Qu4dTableGen
ninja qu4d-mlir-lib
ninja qu4d-opt
```

## Usage

### Using qu4d-opt

```bash
# Convert Linalg to qu4d and lower to SCF
qu4d-opt mm_linalg.mlir \
  -qu4d-from-linalg \
  -qu4d-lower-matmul4x4-scf \
  -canonicalize \
  -o mm_qu4d_lowered.mlir

# Convert lowered mlir to LLVM dialect
mlir-opt mm_qu4d_lowered.mlir -convert-scf-to-cf \
-convert-to-llvm -reconcile-unrealized-casts -o mm_llvm.mlir

# Translate to LLVM IR
mlir-translate \
  --mlir-to-llvmir mm_llvm.mlir -o mm.ll

opt -load-pass-plugin=/path/to/build/MlirToMatMul/MlirToMatMul.dylib \
 -passes=mlir2matmul -o mm_output.ll -S mm.ll
```

### Canonicalize LLVM IR With -mlir


### Example MLIR Input

```mlir
func.func @matmul_4x4(%a: memref<4x4xf32>, %b: memref<4x4xf32>, %c: memref<4x4xf32>) {
  linalg.matmul ins(%a, %b: memref<4x4xf32>, memref<4x4xf32>)
                outs(%c: memref<4x4xf32>)
  return
}
```

After `qu4d-from-linalg`:

```mlir
func.func @matmul_4x4(%a: memref<4x4xf32>, %b: memref<4x4xf32>, %c: memref<4x4xf32>) {
  qu4d.matmul_4x4 %a, %b, %c : memref<4x4xf32>, memref<4x4xf32>, memref<4x4xf32>
  return
}
```

After `lower-matmul-4x4`:

```mlir
func.func @matmul_4x4(%a: memref<4x4xf32>, %b: memref<4x4xf32>, %c: memref<4x4xf32>) {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  %cf0 = arith.constant 0.0 : f32
  
  scf.for %i = %c0 to %c4 step %c1 iter_args(%arg0 = %cf0) -> (f32) {
    scf.for %j = %c0 to %c4 step %c1 {
      scf.for %k = %c0 to %c4 step %c1 {
        // ... matrix multiplication loop body
      }
    }
  }
  return
}
```

## Project Layout
```text
qu4d-mlir/
├── include/qu4d/        # dialect + pass headers
├── lib/                 # dialect + pass implementations
├── tools/qu4d-opt/      # opt-like driver
└── test/qu4d/           # *.mlir / *.ll tests
```

## Key Components

### Qu4dFromLinalg Transform

**File**: `lib/Transforms/Qu4dFromLinalg.cpp`

Converts high-level Linalg operations to specialized qu4d operations.

### LowerMatmul4x4 Transform

**File**: `lib/Transforms/LowerMatmul4x4.cpp`

Lowers qu4d operations to scf loops.

### Operation Verifiers

**File**: `lib/IR/Qu4dOps.cpp`

Custom verifiers ensure correctness.

## Testing and Performance Measurement

Test files are located in `test/qu4d/`:

Include:
- `mm_linalg.mlir`: Input file, using linalg.matmul ops
- `mm_qu4d_lowered.mlir`: After lowering to qu4d.matmul and then to SCF
- `qu4d_matmul4x4_canonicalize.mlir`: Test file for qu4d cananicalization
- `qu4d_matmul4x4_verify.mlir`: Test file for verification regarding memory shapes and alias
- `bench_mat4x4.cpp`: cpp for integration into Clang to run performance benchmark and functionality correctness check
- `mat4x4_16acc_test.cpp`: cpp for generating the optimized kernel (see command below)

Tests are run manually:

```bash
# Check with qu4d-opt
qu4d-opt qu4d_matmul4x4_verify.mlir -verify-diagnostics
qu4d-opt -canonicalize qu4d_matmul4x4_canonicalize.mlir | FileCheck qu4d_matmul4x4_canonicalize.mlir
```

### Generate Vectorized Kernel
```bash
# Generate IR for kernel
clang++ -O3 -ffast-math -std=c++17 -emit-llvm  -S mat4x4_16acc_test.cpp -o mat4x4_O3.ll

# Link with output from mlir2matmul pass (last step of our pipeline)
llvm-link mm_output.ll mat4x4_O3.ll -o merged_O3.ll

# Generate object file
clang++ -O3 -ffast-math merged_O3.ll bench_mat4x4.cpp -o test_merge_O3

# Optionally, instead of llvm-link and then clang, compile and link directly with clang
clang++ -O3 -ffast-math mat4x4_16acc_test.cpp mm_output.ll bench_mat4x4.cpp -o test_merge_O3

# Run benchmark (also runs a small test for correctness)
./test_merge_O3
```

### Sample output

Performance is measured in GFLOPs/second
```bash
mat4x4_16acc_kernel: 79.40 GFLOP/s  (iters=50000000)
```

## References

- [MLIR Documentation](https://mlir.llvm.org/)
- [MLIR Dialect Development](https://mlir.llvm.org/docs/Tutorials/DefiningDialects/Operations/)
- [MLIR Pattern Rewriting](https://mlir.llvm.org/docs/PatternRewriter/)
- [LLVM Documentation](https://llvm.org/docs/)
