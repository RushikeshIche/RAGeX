#include "inference.h"
#include <string.h>
#include <math.h>

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

void forward_word_embeddings(const MiniLM* model, const int* token_ids, int seq_len, float* output) {
    const Tensor* wte = &model->embeddings.word_embeddings;
    
    // In our optimized format, word embeddings are stored as INT8
    if (wte->dtype == DTYPE_INT8) {
        const int8_t* weight_data = (const int8_t*)wte->data;
        float scale = wte->scale;
        
        for (int i = 0; i < seq_len; ++i) {
            int token_id = token_ids[i];
            
            // Safety bounds check
            if (token_id < 0 || token_id >= VOCAB_SIZE) {
                token_id = 0; 
            }
            
            const int8_t* in_row = weight_data + (token_id * HIDDEN_SIZE);
            float* out_row = output + (i * HIDDEN_SIZE);
            
            // Dequantize the 8-bit integer into a 32-bit float on the fly
            for (int d = 0; d < HIDDEN_SIZE; ++d) {
                out_row[d] = ((float)in_row[d]) * scale;
            }
        }
    } 
    // Fallback if we export them as F32
    else if (wte->dtype == DTYPE_F32) {
        const float* weight_data = (const float*)wte->data;
        
        for (int i = 0; i < seq_len; ++i) {
            int token_id = token_ids[i];
            if (token_id < 0 || token_id >= VOCAB_SIZE) {
                token_id = 0; 
            }
            
            const float* in_row = weight_data + (token_id * HIDDEN_SIZE);
            float* out_row = output + (i * HIDDEN_SIZE);
            memcpy(out_row, in_row, HIDDEN_SIZE * sizeof(float));
        }
    }
}

void add_positional_embeddings(const MiniLM* model, int seq_len, float* output) {
    const Tensor* pos_emb = &model->embeddings.position_embeddings;
    const Tensor* type_emb = &model->embeddings.token_type_embeddings;
    
    // For RAG search, we only use one sentence at a time, so token type is always 0
    int type_id = 0; 

    // Add Positional Embeddings
    if (pos_emb->dtype == DTYPE_INT8) {
        const int8_t* p_data = (const int8_t*)pos_emb->data;
        float p_scale = pos_emb->scale;
        
        for (int i = 0; i < seq_len; ++i) {
            float* out_row = output + (i * HIDDEN_SIZE);
            int pos_idx = (i >= MAX_SEQ_LEN) ? (MAX_SEQ_LEN - 1) : i; // Cap at max length
            const int8_t* pos_row = p_data + (pos_idx * HIDDEN_SIZE);
            
            for (int d = 0; d < HIDDEN_SIZE; ++d) {
                out_row[d] += ((float)pos_row[d]) * p_scale;
            }
        }
    } else if (pos_emb->dtype == DTYPE_F32) {
        const float* p_data = (const float*)pos_emb->data;
        for (int i = 0; i < seq_len; ++i) {
            float* out_row = output + (i * HIDDEN_SIZE);
            int pos_idx = (i >= MAX_SEQ_LEN) ? (MAX_SEQ_LEN - 1) : i;
            const float* pos_row = p_data + (pos_idx * HIDDEN_SIZE);
            
            for (int d = 0; d < HIDDEN_SIZE; ++d) {
                out_row[d] += pos_row[d];
            }
        }
    }
    
    // Add Token Type Embeddings
    if (type_emb->dtype == DTYPE_INT8) {
        const int8_t* t_data = (const int8_t*)type_emb->data;
        float t_scale = type_emb->scale;
        const int8_t* type_row = t_data + (type_id * HIDDEN_SIZE);
        
        for (int i = 0; i < seq_len; ++i) {
            float* out_row = output + (i * HIDDEN_SIZE);
            for (int d = 0; d < HIDDEN_SIZE; ++d) {
                out_row[d] += ((float)type_row[d]) * t_scale;
            }
        }
    } else if (type_emb->dtype == DTYPE_F32) {
        const float* t_data = (const float*)type_emb->data;
        const float* type_row = t_data + (type_id * HIDDEN_SIZE);
        
        for (int i = 0; i < seq_len; ++i) {
            float* out_row = output + (i * HIDDEN_SIZE);
            for (int d = 0; d < HIDDEN_SIZE; ++d) {
                out_row[d] += type_row[d];
            }
        }
    }
}

// Helper function: PyTorch Linear layer math (Y = X @ W^T + b)
static void linear_forward(int M, int K, int N, const float* X, const Tensor* W, const Tensor* b, float* Y) {
    int is_int8 = (W->dtype == DTYPE_INT8);
    const int8_t* W_int8 = (const int8_t*)W->data;
    const float* W_f32 = (const float*)W->data;
    float w_scale = is_int8 ? W->scale : 1.0f;
    const float* bias = (b && b->data) ? (const float*)b->data : NULL;
    
    // Standard unoptimized matrix multiplication
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = bias ? bias[j] : 0.0f;
            for (int k = 0; k < K; ++k) {
                float w_val = is_int8 ? ((float)W_int8[j * K + k] * w_scale) : W_f32[j * K + k];
                sum += X[i * K + k] * w_val;
            }
            Y[i * N + j] = sum;
        }
    }
}

void forward_attention(const Attention* attn, int seq_len, const float* x, float* output, BumpAllocator* mem) {
    int hidden_size = HIDDEN_SIZE;
    int num_heads = NUM_ATTENTION_HEADS;
    int head_dim = hidden_size / num_heads;
    
    // Allocate fast temporary memory for Q, K, V
    float* Q = (float*)bump_alloc(mem, seq_len * hidden_size * sizeof(float));
    float* K = (float*)bump_alloc(mem, seq_len * hidden_size * sizeof(float));
    float* V = (float*)bump_alloc(mem, seq_len * hidden_size * sizeof(float));
    float* context = (float*)bump_alloc(mem, seq_len * hidden_size * sizeof(float));
    float* scores = (float*)bump_alloc(mem, seq_len * sizeof(float));
    
    // Project Input -> Query, Key, Value
    linear_forward(seq_len, hidden_size, hidden_size, x, &attn->self.query.weight, &attn->self.query.bias, Q);
    linear_forward(seq_len, hidden_size, hidden_size, x, &attn->self.key.weight, &attn->self.key.bias, K);
    linear_forward(seq_len, hidden_size, hidden_size, x, &attn->self.value.weight, &attn->self.value.bias, V);
    
    // Multi-Head Self-Attention Algorithm
    float scale = 1.0f / sqrtf((float)head_dim);
    
    for (int h = 0; h < num_heads; ++h) {
        for (int i = 0; i < seq_len; ++i) { // For every word (query)
            float* q_vec = Q + (i * hidden_size + h * head_dim);
            
            // Calculate Attention Scores (Q dotted with K)
            float max_score = -1e9f;
            for (int j = 0; j < seq_len; ++j) { // Compare against every other word (key)
                float* k_vec = K + (j * hidden_size + h * head_dim);
                float dot = 0.0f;
                
#ifdef __ARM_NEON
                float32x4_t sum_vec = vdupq_n_f32(0.0f);
                for (int d = 0; d < head_dim; d += 4) {
                    float32x4_t q_v = vld1q_f32(&q_vec[d]);
                    float32x4_t k_v = vld1q_f32(&k_vec[d]);
                    sum_vec = vmlaq_f32(sum_vec, q_v, k_v);
                }
                float sum_arr[4];
                vst1q_f32(sum_arr, sum_vec);
                dot = sum_arr[0] + sum_arr[1] + sum_arr[2] + sum_arr[3];
#else
                for (int d = 0; d < head_dim; ++d) {
                    dot += q_vec[d] * k_vec[d];
                }
#endif
                
                dot *= scale; // Scale down so numbers don't explode
                scores[j] = dot;
                if (dot > max_score) max_score = dot;
            }
            
            // Softmax (Convert scores to probabilities summing to 1.0)
            float sum_exp = 0.0f;
            for (int j = 0; j < seq_len; ++j) {
                scores[j] = expf(scores[j] - max_score);
                sum_exp += scores[j];
            }
            for (int j = 0; j < seq_len; ++j) {
                scores[j] /= sum_exp;
            }
            
            // Weighted sum of Values (V)
            float* ctx_vec = context + (i * hidden_size + h * head_dim);
            for (int d = 0; d < head_dim; ++d) ctx_vec[d] = 0.0f;
            
            for (int j = 0; j < seq_len; ++j) {
                float* v_vec = V + (j * hidden_size + h * head_dim);
                float s = scores[j];
                
#ifdef __ARM_NEON
                float32x4_t s_vec = vdupq_n_f32(s);
                for (int d = 0; d < head_dim; d += 4) {
                    float32x4_t ctx_v = vld1q_f32(&ctx_vec[d]);
                    float32x4_t v_v = vld1q_f32(&v_vec[d]);
                    ctx_v = vmlaq_f32(ctx_v, s_vec, v_v);
                    vst1q_f32(&ctx_vec[d], ctx_v);
                }
#else
                for (int d = 0; d < head_dim; ++d) {
                    ctx_vec[d] += s * v_vec[d]; // Mix meaning based on attention probability!
                }
#endif
            }
        }
    }
    
    // Final Output Projection
    linear_forward(seq_len, hidden_size, hidden_size, context, &attn->output.dense.weight, &attn->output.dense.bias, output);
}

// GELU activation: the version of ReLU used inside all BERT style transformers
// Formula: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
static inline float gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}

void forward_ffn(const Intermediate* intermediate, const OutputBlock* output_block,
                 int seq_len, const float* x, float* out, BumpAllocator* mem) {
    // Allocate a temporary buffer in the bump arena for the expanded intermediate output
    float* mid = (float*)bump_alloc(mem, seq_len * INTERMEDIATE_SIZE * sizeof(float));

    linear_forward(seq_len, HIDDEN_SIZE, INTERMEDIATE_SIZE, x, &intermediate->dense.weight, &intermediate->dense.bias, mid);

    int total_mid = seq_len * INTERMEDIATE_SIZE;

#ifdef __ARM_NEON

    int limit = total_mid - (total_mid % 4);
    for (int i = 0; i < limit; i += 4) {
        // NEON has no built-in tanh, so we call gelu() individually but unroll 4 at a time
        // (compiler will still vectorize loads/stores around this)
        mid[i]     = gelu(mid[i]);
        mid[i + 1] = gelu(mid[i + 1]);
        mid[i + 2] = gelu(mid[i + 2]);
        mid[i + 3] = gelu(mid[i + 3]);
    }
    for (int i = limit; i < total_mid; ++i) {
        mid[i] = gelu(mid[i]);
    }
#else
    for (int i = 0; i < total_mid; ++i) {
        mid[i] = gelu(mid[i]);
    }
#endif

    linear_forward(seq_len, INTERMEDIATE_SIZE, HIDDEN_SIZE, mid, &output_block->dense.weight, &output_block->dense.bias, out);
}

void layer_norm(const LayerNorm* ln, float* x, int seq_len) {
    const float eps = 1e-12f; // Small epsilon to prevent division by zero
    const float* weight = (const float*)ln->weight.data;
    const float* bias   = (const float*)ln->bias.data;
    
    for (int i = 0; i < seq_len; ++i) {
        float* row = x + (i * HIDDEN_SIZE);
        
        // Compute mean of this token's 384 values
        float mean = 0.0f;
        for (int d = 0; d < HIDDEN_SIZE; ++d) {
            mean += row[d];
        }
        mean /= (float)HIDDEN_SIZE;
        
        // Compute variance
        float var = 0.0f;
        for (int d = 0; d < HIDDEN_SIZE; ++d) {
            float diff = row[d] - mean;
            var += diff * diff;
        }
        var /= (float)HIDDEN_SIZE;
        
        // Normalize and apply learned scale/shift
        float inv_std = 1.0f / sqrtf(var + eps);
        
#ifdef __ARM_NEON
        float32x4_t mean_v = vdupq_n_f32(mean);
        float32x4_t istd_v = vdupq_n_f32(inv_std);
        int limit = HIDDEN_SIZE - (HIDDEN_SIZE % 4);
        
        for (int d = 0; d < limit; d += 4) {
            float32x4_t v = vld1q_f32(&row[d]);
            float32x4_t w = vld1q_f32(&weight[d]);
            float32x4_t b = vld1q_f32(&bias[d]);
            // (x - mean) / std * weight + bias
            v = vmulq_f32(vsubq_f32(v, mean_v), istd_v);
            v = vmlaq_f32(b, v, w);
            vst1q_f32(&row[d], v);
        }
        for (int d = limit; d < HIDDEN_SIZE; ++d) {
            row[d] = ((row[d] - mean) * inv_std) * weight[d] + bias[d];
        }
#else
        for (int d = 0; d < HIDDEN_SIZE; ++d) {
            row[d] = ((row[d] - mean) * inv_std) * weight[d] + bias[d];
        }
#endif
    }
}

