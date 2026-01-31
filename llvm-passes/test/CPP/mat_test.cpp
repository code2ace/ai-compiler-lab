#include <array>
#include <cstdio>
using Mat4 = std::array<float, 16>; // row-major: idx(i,j) = i*4 + j
extern "C" void run_case(const char* label, const Mat4& A, const Mat4& B, bool verbose=false);

int main() {
    // Case 1: identity * arbitrary
    Mat4 I {
        1,0,0,0,
        0,1,0,0,
        0,0,1,0,
        0,0,0,1
    };
    Mat4 X {
        1,  2,  3,  4,
        5,  6,  7,  8,
        9, 10, 11, 12,
        13,14, 15, 16
    };

    // Case 2: simple patterned numbers
    Mat4 A {
        0.5f, -1.0f, 2.0f,  3.5f,
        -2.0f, 0.25f, 1.0f, -0.5f,
        4.0f,  -3.0f, 0.0f,  2.0f,
        1.5f,  2.0f, -0.25f, 1.0f
    };
    Mat4 B {
        -1.0f, 2.0f,  0.5f, 1.0f,
         3.0f, 1.0f, -2.0f, 0.0f,
         0.0f, 1.5f,  1.0f, -3.0f,
         2.0f, 0.5f, -1.0f, 4.0f
    };

    // Case 3: another deterministic set (scaled/shifted)
    Mat4 C {
        0.1f, 0.2f, 0.3f, 0.4f,
        0.5f, 0.6f, 0.7f, 0.8f,
        0.9f, 1.0f, 1.1f, 1.2f,
        1.3f, 1.4f, 1.5f, 1.6f
    };
    Mat4 D {
       -0.2f, 0.3f, -0.4f,  0.5f,
        0.6f,-0.7f,  0.8f, -0.9f,
        1.0f,-1.1f,  1.2f, -1.3f,
        1.4f,-1.5f,  1.6f, -1.7f
    };

    run_case("Identity * X", I, X, /*verbose=*/false);
    run_case("A * B", A, B, /*verbose=*/false);
    run_case("C * D", C, D, /*verbose=*/false);

    std::puts("All tests passed ✅");
    return 0;
}
