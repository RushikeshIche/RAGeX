#include "inference.h"
#include <string.h>

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
