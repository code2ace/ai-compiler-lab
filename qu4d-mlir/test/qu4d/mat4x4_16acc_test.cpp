// mat4x4_16acc_test.cpp
// Build (native run):
//   clang++ -O2 -std=c++17 mat4x4_16acc_test.cpp -o mat4x4_test && ./mat4x4_test
//
// Build IR for your pass (keeps fmul+fadd separate; no auto-FMA):
//   clang++ -O0 -ffp-contract=off -S -emit-llvm mat4x4_16acc_test.cpp -o mat4x4_16acc_test.ll
//   # Then run your pass on the .ll and compare outputs.
//
// Optional: IR with nice canonical loops for matching:
//   clang++ -O0 -ffp-contract=off -S -emit-llvm mat4x4_16acc_test.cpp -o - |
//   opt -S -passes='mem2reg,instcombine,simplifycfg,loop-simplify,lcssa,indvars' > dev.ll

#include <cstdio>
#include <cmath>
#include <algorithm>
#include <array>
#include <cassert>

using Mat4 = std::array<float, 16>; // row-major: idx(i,j) = i*4 + j

static inline void mat4x4_ref_triple(const float* A, const float* B, float* C) {
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            float cij = 0.0f;
            for (int k = 0; k < 4; ++k) {
                cij += A[i*4 + k] * B[k*4 + j];
            }
            C[i*4 + j] = cij;
        }
    }
}

// Outer-product micro-kernel with 16 explicit accumulators.
// Updates c00..c33 every k-iteration. Row-major A,B,C.
extern "C" void mat4x4_16acc_kernel(const float* A, const float* B, float* C) {
    float c00=0.f, c01=0.f, c02=0.f, c03=0.f;
    float c10=0.f, c11=0.f, c12=0.f, c13=0.f;
    float c20=0.f, c21=0.f, c22=0.f, c23=0.f;
    float c30=0.f, c31=0.f, c32=0.f, c33=0.f;

    for (int k = 0; k < 4; ++k) {
        // A column (kth) for rows 0..3
        float a0 = A[0*4 + k];
        float a1 = A[1*4 + k];
        float a2 = A[2*4 + k];
        float a3 = A[3*4 + k];
        // B row (kth) for cols 0..3
        float b0 = B[k*4 + 0];
        float b1 = B[k*4 + 1];
        float b2 = B[k*4 + 2];
        float b3 = B[k*4 + 3];

        // 16 scalar MACs (fmul+fadd) — perfect for your pass to turn into vFMA
        c00 += a0 * b0;  c01 += a0 * b1;  c02 += a0 * b2;  c03 += a0 * b3;
        c10 += a1 * b0;  c11 += a1 * b1;  c12 += a1 * b2;  c13 += a1 * b3;
        c20 += a2 * b0;  c21 += a2 * b1;  c22 += a2 * b2;  c23 += a2 * b3;
        c30 += a3 * b0;  c31 += a3 * b1;  c32 += a3 * b2;  c33 += a3 * b3;
    }

    C[0*4 + 0] = c00; C[0*4 + 1] = c01; C[0*4 + 2] = c02; C[0*4 + 3] = c03;
    C[1*4 + 0] = c10; C[1*4 + 1] = c11; C[1*4 + 2] = c12; C[1*4 + 3] = c13;
    C[2*4 + 0] = c20; C[2*4 + 1] = c21; C[2*4 + 2] = c22; C[2*4 + 3] = c23;
    C[3*4 + 0] = c30; C[3*4 + 1] = c31; C[3*4 + 2] = c32; C[3*4 + 3] = c33;
}

static void print_mat(const char* name, const float* M) {
    std::printf("%s =\n", name);
    for (int i = 0; i < 4; ++i) {
        std::printf("  ");
        for (int j = 0; j < 4; ++j) std::printf("%8.4f ", M[i*4 + j]);
        std::printf("\n");
    }
}

static float max_abs_diff(const float* X, const float* Y) {
    float m = 0.f;
    for (int i = 0; i < 16; ++i) m = std::max(m, std::fabs(X[i] - Y[i]));
    return m;
}

static bool nearly_equal_mat(const float* X, const float* Y, float eps = 1e-5f) {
    return max_abs_diff(X, Y) <= eps;
}

extern "C" void run_case(const char* label, const Mat4& A, const Mat4& B, bool verbose=false) {
    Mat4 C_ref{}, C_ker{};
    mat4x4_ref_triple(A.data(), B.data(), C_ref.data());
    mat4x4_16acc_kernel(A.data(), B.data(), C_ker.data());

    float mad = max_abs_diff(C_ref.data(), C_ker.data());
    bool ok = nearly_equal_mat(C_ref.data(), C_ker.data());
    std::printf("[%s] max|diff| = %.8f  => %s\n", label, mad, ok ? "OK" : "MISMATCH");
    if (verbose || !ok) {
        print_mat("A", A.data());
        print_mat("B", B.data());
        print_mat("C_ref", C_ref.data());
        print_mat("C_ker", C_ker.data());
    }
    if (!ok) std::exit(1);
}

 