// RUN: qu4d-opt %s -verify-diagnostics

// Expected behavior: this one is OK (4x4 memrefs, no alias).
func.func @ok_memref(%A: memref<4x4xf32>,
                     %B: memref<4x4xf32>,
                     %C: memref<4x4xf32>) {
  qu4d.matmul_4x4 %A, %B, %C
    : memref<4x4xf32>, memref<4x4xf32>, memref<4x4xf32>
  func.return
}

// -----

// This should fail the shape/type check.
func.func @bad_shape(%A: memref<5x4xf32>,
                     %B: memref<4x4xf32>,
                     %C: memref<4x4xf32>) {
// expected-error @+1 {{operands/results must be memref<4x4xf32>/tensor<4x4xf32> or vector<4xf32>}}
  qu4d.matmul_4x4 %A, %B, %C
    : memref<5x4xf32>, memref<4x4xf32>, memref<4x4xf32>
  func.return
}

// -----

// This should fail the alias check (A and B are the same buffer).
func.func @alias_AB(%A: memref<4x4xf32>,
                    %C: memref<4x4xf32>) {
  // Pass %A for both A and B.
  // expected-error @+1 {{operand A and B should not alias}}
  qu4d.matmul_4x4 %A, %A, %C
    : memref<4x4xf32>, memref<4x4xf32>, memref<4x4xf32>
  func.return
}

func.func @alias_via_cast(%A: memref<4x4xf32>,
                          %C: memref<4x4xf32>) {
  %A_cast = memref.cast %A : memref<4x4xf32> to memref<4x4xf32>
  // expected-error @+1 {{operand A and B should not alias}}
  qu4d.matmul_4x4 %A, %A_cast, %C
    : memref<4x4xf32>, memref<4x4xf32>, memref<4x4xf32>
  func.return
}