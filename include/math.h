#ifndef RAGEX_MATH_H
#define RAGEX_MATH_H

#include "types.h"

/**
 * Baseline Matrix Multiplication (GEMM) for fp32.
 * Computes: C = alpha * A * B + beta * C
 * 
 * A is shape (M x K)
 * B is shape (K x N)
 * C is shape (M x N)
 * All matrices are assumed to be row-major.
 */
void gemm_f32_baseline(int M, int N, int K, float alpha, const float* A, const float* B, float beta, float* C);

#ifdef __ARM_NEON
#include <arm_neon.h>
/**
 * ARM NEON accelerated Matrix Multiplication (GEMM) for fp32.
 */
void gemm_f32_neon(int M, int N, int K, float alpha, const float* A, const float* B, float beta, float* C);
#endif

#endif // RAGEX_MATH_H
