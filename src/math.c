#include "math.h"

void gemm_f32_baseline(int M, int N, int K, float alpha, const float* A, const float* B, float beta, float* C) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                // A: row-major (M x K)
                // B: row-major (K x N)
                sum += A[i * K + k] * B[k * N + j];
            }
            
            int c_idx = i * N + j;
            if (beta == 0.0f) {
                C[c_idx] = alpha * sum;
            } else {
                C[c_idx] = alpha * sum + beta * C[c_idx];
            }
        }
    }
}
