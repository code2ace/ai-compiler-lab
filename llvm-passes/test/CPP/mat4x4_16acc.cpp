// 4x4 matmul with 16 explicit accumulators: C = A * B (row-major)
// The inner K-loop updates c00..c33 every iteration.
// This shape makes the 16 fmul+fadd pairs visible in IR.
extern "C" void mat4x4_mul_16acc(const float* A, const float* B, float* C) {
    // Initialize 16 accumulators
    float c00=0.f, c01=0.f, c02=0.f, c03=0.f;
    float c10=0.f, c11=0.f, c12=0.f, c13=0.f;
    float c20=0.f, c21=0.f, c22=0.f, c23=0.f;
    float c30=0.f, c31=0.f, c32=0.f, c33=0.f;

    for (int k = 0; k < 4; ++k) {
        // Load one column of B (kth row) and one column of A (kth col)
        float a0 = A[0*4 + k];
        float a1 = A[1*4 + k];
        float a2 = A[2*4 + k];
        float a3 = A[3*4 + k];

        float b0 = B[k*4 + 0];
        float b1 = B[k*4 + 1];
        float b2 = B[k*4 + 2];
        float b3 = B[k*4 + 3];

        // 16 scalar updates (each is fmul + fadd in IR when contraction is off)
        c00 += a0 * b0; c01 += a0 * b1; c02 += a0 * b2; c03 += a0 * b3;
        c10 += a1 * b0; c11 += a1 * b1; c12 += a1 * b2; c13 += a1 * b3;
        c20 += a2 * b0; c21 += a2 * b1; c22 += a2 * b2; c23 += a2 * b3;
        c30 += a3 * b0; c31 += a3 * b1; c32 += a3 * b2; c33 += a3 * b3;
    }

    // Store results to C
    C[0*4 + 0] = c00; C[0*4 + 1] = c01; C[0*4 + 2] = c02; C[0*4 + 3] = c03;
    C[1*4 + 0] = c10; C[1*4 + 1] = c11; C[1*4 + 2] = c12; C[1*4 + 3] = c13;
    C[2*4 + 0] = c20; C[2*4 + 1] = c21; C[2*4 + 2] = c22; C[2*4 + 3] = c23;
    C[3*4 + 0] = c30; C[3*4 + 1] = c31; C[3*4 + 2] = c32; C[3*4 + 3] = c33;
}