include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinTypes.td"
include "Qu4dDialect.td"

// A minimal elementwise ReLU op for warm-up
def Qu4d_ReluOp : Op<Qu4d_Dialect, "relu"> {
    let summary = "ReLU (max(x, 0)) for f32 or vector<f32>";
    let arguments = (ins AnyType:$x);
    let results = (outs AnyType:$y);
    let assemblyFormat = "$x attr-dict `:` type($x) `->` type($y)";
}

// Skeleton for a shape-specialized 4x4 matmul microkernel op
// Add layout/align/contiguity attributes later.
def Qu4d_Matmul4x4Op : Op<Qu4d_Dialect, "matmul_4x4"> {
    let summary = "Fixed 4x4 microkernel matmul with verified shapes";
    let arguments = (ins AnyMemRef:$a, AnyMemRef:$b, AnyMemRef:$c);

    let results = (outs);
    let assemblyFormat =
    "$a `,` $b `,` $c attr-dict `:` type($a) `,` type($b) `,` type($c)";
   
    let hasVerifier = 1; // providing a custom C++ verifier
    let hasCanonicalizer = 1;
}
