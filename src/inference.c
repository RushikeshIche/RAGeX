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
