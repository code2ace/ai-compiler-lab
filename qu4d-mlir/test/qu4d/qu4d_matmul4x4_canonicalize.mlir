// RUN: qu4d-opt %s -canonicalize | FileCheck %s

// CHECK-LABEL: func.func @fold_unused_C
// CHECK-NOT: qu4d.matmul_4x4
// CHECK: return
func.func @fold_unused_C(%A: memref<4x4xf32>, %B: memref<4x4xf32>) {
    // C is only used by qu4d.matmul_4x4 → your canonicalization should erase it.
    %C = memref.alloc() : memref<4x4xf32>
    qu4d.matmul_4x4 %A, %B, %C
        : memref<4x4xf32>, memref<4x4xf32>, memref<4x4xf32>
    return
}

// CHECK-LABEL: func.func @dont_fold_when_C_is_used
// CHECK:       qu4d.matmul_4x4
func.func @dont_fold_when_C_is_used(%A: memref<4x4xf32>,
                                    %B: memref<4x4xf32>,
                                    %C: memref<4x4xf32>) -> f32 {
    qu4d.matmul_4x4 %A, %B, %C
    : memref<4x4xf32>, memref<4x4xf32>, memref<4x4xf32>

    %c0 = arith.constant 0 : index
    %v = memref.load %C[%c0, %c0] : memref<4x4xf32>
    return %v : f32
}
