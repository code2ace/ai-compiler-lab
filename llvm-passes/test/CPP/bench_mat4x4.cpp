#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <array>

using Mat4 = std::array<float, 16>;

extern "C" void mat4x4_16acc_kernel(const float* A, const float* B, float* C); // your kernel
static void ref_triple(const float* A, const float* B, float* C) {
    for (int i=0;i<4;i++)
      for (int j=0;j<4;j++) {
        float s=0.f;
        for (int k=0;k<4;k++) s += A[i*4+k]*B[k*4+j];
        C[i*4+j]=s;
      }
}

// (Optional) scalar baseline version (no vector intrinsics), used if you compile this file alone
extern "C" void mat4x4_scalar_baseline(const float* A, const float* B, float* C) {
    ref_triple(A,B,C);
}

static float max_abs_diff(const float* X, const float* Y) {
    float m=0.f; for (int i=0;i<16;i++) m = std::max(m, std::fabs(X[i]-Y[i])); return m;
}

// Prevent inlining
template <typename Fn>
__attribute__((noinline)) static void run_many(Fn f, const float* A, const float* B, float* C, int iters) {
    for (int t=0; t<iters; ++t) f(A,B,C);
}

// Return gflops/s. Each 4x4 matmul = 2*M*N*K = 128 FLOPs (counting FMA as 2 flops)
template <typename Fn>
static double bench(Fn f, const float* A, const float* B, float* C, int iters) {
    using clk = std::chrono::high_resolution_clock;
    auto t0 = clk::now();
    run_many(f, A, B, C, iters);
    auto t1 = clk::now();
    std::chrono::duration<double> dt = t1 - t0;
    double flops = 128.0 * iters;
    return (flops / dt.count()) / 1e9; // GFLOP/s
}

int main(int argc, char** argv) {
    int iters = (argc>1) ? std::atoi(argv[1]) : 50'000'000; // big loop for stable timing
    // deterministic inputs
    Mat4 A {0.5f,-1,2,3.5,-2,0.25f,1,-0.5,4,-3,0,2,1.5,2,-0.25f,1};
    Mat4 B {-1,2,0.5f,1,3,1,-2,0,0,1.5f,1,-3,2,0.5f,-1,4};
    Mat4 Cref{}, Ctest{};

    // sanity check kernel once
    ref_triple(A.data(), B.data(), Cref.data());
    mat4x4_16acc_kernel(A.data(), B.data(), Ctest.data());
    float mad = max_abs_diff(Cref.data(), Ctest.data());
    if (mad > 1e-5f) {
        std::printf("Sanity check FAILED, max_abs_diff=%.8f\n", mad);
        return 1;
    }

    // Warm-up
    for (int i=0;i<20000;i++) mat4x4_16acc_kernel(A.data(), B.data(), Ctest.data());

    // Timed
    double gflops = bench(mat4x4_16acc_kernel, A.data(), B.data(), Ctest.data(), iters);
    std::printf("mat4x4_16acc_kernel: %.2f GFLOP/s  (iters=%d)\n", gflops, iters);
    return 0;
}
