module {
    func.func @test(%A: memref<4x4xf32>, %B: memref<4x4xf32>, %C: memref<4x4xf32>) {
        qu4d.matmul_4x4 %A, %B, %C
            : memref<4x4xf32>, memref<4x4xf32>, memref<4x4xf32>
        func.return
    }

    func.func @test_fold_C(%A: memref<4x4xf32>, %B: memref<4x4xf32>) {
        %C = memref.alloc() : memref<4x4xf32>
        qu4d.matmul_4x4 %A, %B, %C
            : memref<4x4xf32>, memref<4x4xf32>, memref<4x4xf32>
        func.return
    }
}

