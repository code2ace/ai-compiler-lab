// RUN: qu4d-opt %s -qu4d-from-linalg | FileCheck %s

func.func @matmul_4x4(%A: memref<4x4xf32>,
                      %B: memref<4x4xf32>,
                      %C: memref<4x4xf32>) {
  // CHECK-LABEL: func.func @matmul_4x4
  // CHECK:       qu4d.matmul_4x4 %arg0, %arg1, %arg2
  // CHECK-NOT:   linalg.matmul
  linalg.matmul
    ins(%A, %B : memref<4x4xf32>, memref<4x4xf32>)
    outs(%C : memref<4x4xf32>)
  return
}

// -----

// Non-4x4 matmul shouldn't be touched.
// CHECK-LABEL: func.func @matmul_8x8
// CHECK:       linalg.matmul
func.func @matmul_8x8(%A: memref<8x8xf32>,
                      %B: memref<8x8xf32>,
                      %C: memref<8x8xf32>) {
  linalg.matmul
    ins(%A, %B : memref<8x8xf32>, memref<8x8xf32>)
    outs(%C : memref<8x8xf32>)
  return
}