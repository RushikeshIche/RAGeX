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

void gemm_f32_int8_baseline(int M, int N, int K, float alpha, const float* A, const int8_t* B, float b_scale, float beta, float* C) {
    float inv_scale = 1.0f / b_scale; // Convert scale back
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * (float)B[k * N + j];
            }
            sum *= inv_scale; // Apply dequantization scale
            
            int c_idx = i * N + j;
            if (beta == 0.0f) {
                C[c_idx] = alpha * sum;
            } else {
                C[c_idx] = alpha * sum + beta * C[c_idx];
            }
        }
    }
}

#ifdef __ARM_NEON
void gemm_f32_neon(int M, int N, int K, float alpha, const float* A, const float* B, float beta, float* C) {
    int n_limit = N - (N % 4);
    
    for (int i = 0; i < M; ++i) {
        // NEON accelerated loop (processes 4 columns at a time)
        for (int j = 0; j < n_limit; j += 4) {
            float32x4_t sum_vec = vdupq_n_f32(0.0f);
            
            // Accumulate outer products to avoid non-contiguous memory access on B
            for (int k = 0; k < K; ++k) {
                float32x4_t a_vec = vdupq_n_f32(A[i * K + k]);
                float32x4_t b_vec = vld1q_f32(&B[k * N + j]);
                sum_vec = vmlaq_f32(sum_vec, a_vec, b_vec);
            }
            
            int c_idx = i * N + j;
            float32x4_t alpha_vec = vdupq_n_f32(alpha);
            float32x4_t res_vec = vmulq_f32(sum_vec, alpha_vec);
            
            if (beta != 0.0f) {
                float32x4_t beta_vec = vdupq_n_f32(beta);
                float32x4_t c_vec = vld1q_f32(&C[c_idx]);
                res_vec = vmlaq_f32(res_vec, beta_vec, c_vec);
            }
            
            vst1q_f32(&C[c_idx], res_vec);
        }
        
        // Handle remainders (if N is not a multiple of 4)
        for (int j = n_limit; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
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

void gemm_f32_int8_neon(int M, int N, int K, float alpha, const float* A, const int8_t* B, float b_scale, float beta, float* C) {
    float inv_scale = 1.0f / b_scale;
    int n_limit = N - (N % 4);
    
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < n_limit; j += 4) {
            float32x4_t sum_vec = vdupq_n_f32(0.0f);
            
            for (int k = 0; k < K; ++k) {
                float32x4_t a_vec = vdupq_n_f32(A[i * K + k]);
                
                // Read 4 int8 bytes and cast to float
                int b_idx = k * N + j;
                float b_arr[4] = { (float)B[b_idx], (float)B[b_idx+1], (float)B[b_idx+2], (float)B[b_idx+3] };
                float32x4_t b_vec = vld1q_f32(b_arr);
                
                sum_vec = vmlaq_f32(sum_vec, a_vec, b_vec);
            }
            
            // Apply scale
            float32x4_t scale_vec = vdupq_n_f32(inv_scale);
            sum_vec = vmulq_f32(sum_vec, scale_vec);
            
            int c_idx = i * N + j;
            float32x4_t alpha_vec = vdupq_n_f32(alpha);
            float32x4_t res_vec = vmulq_f32(sum_vec, alpha_vec);
            
            if (beta != 0.0f) {
                float32x4_t beta_vec = vdupq_n_f32(beta);
                float32x4_t c_vec = vld1q_f32(&C[c_idx]);
                res_vec = vmlaq_f32(res_vec, beta_vec, c_vec);
            }
            
            vst1q_f32(&C[c_idx], res_vec);
        }
        
        // Handle remainders (if N is not a multiple of 4)
        for (int j = n_limit; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * (float)B[k * N + j];
            }
            sum *= inv_scale;
            
            int c_idx = i * N + j;
            if (beta == 0.0f) {
                C[c_idx] = alpha * sum;
            } else {
                C[c_idx] = alpha * sum + beta * C[c_idx];
            }
        }
    }
}
#endif

