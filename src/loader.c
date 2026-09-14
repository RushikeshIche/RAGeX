#include "loader.h"
#include <stdio.h>
#include <string.h>

// A simple macro to map the data pointer to the struct field based on the name
#define MAP_TENSOR(NAME, TARGET) \
    if (strcmp(name, NAME) == 0) { \
        TARGET.ndim = ndim; \
        for(int i=0; i<4; ++i) TARGET.shape[i] = (i < ndim) ? shape[i] : 1; \
        TARGET.dtype = (dtype == 1) ? DTYPE_INT8 : DTYPE_F32; \
        TARGET.scale = scale; \
        TARGET.data = (void*)data_ptr; \
        matched = 1; \
    }

int load_minilm_model(const char* filepath, MiniLM* model, MmapFile* mf) {
    if (mmap_file_read(filepath, mf) != 0) {
        printf("Error: Failed to mmap file %s\n", filepath);
        return -1;
    }

    uint8_t* ptr = (uint8_t*)mf->data;
    uint8_t* end = ptr + mf->size;
    int tensors_loaded = 0;

    printf("Loading weights from mapped memory...\n");
    while (ptr < end) {
        // Read string length
        uint32_t name_len = *(uint32_t*)ptr;
        ptr += sizeof(uint32_t);

        // Read string
        char name[256] = {0};
        if (name_len < sizeof(name)) {
            memcpy(name, ptr, name_len);
        }
        ptr += name_len;

        // Read ndim
        uint32_t ndim = *(uint32_t*)ptr;
        ptr += sizeof(uint32_t);

        // Read shape and compute size
        int shape[4] = {1, 1, 1, 1};
        int num_elements = 1;
        for (uint32_t i = 0; i < ndim; ++i) {
            shape[i] = *(uint32_t*)ptr;
            num_elements *= shape[i];
            ptr += sizeof(uint32_t);
        }

        // Read DType (0=F32, 1=INT8)
        uint32_t dtype = *(uint32_t*)ptr;
        ptr += sizeof(uint32_t);

        // Read Scale (if INT8)
        float scale = 1.0f;
        if (dtype == 1) {
            scale = *(float*)ptr;
            ptr += sizeof(float);
        }

        // Now ptr is at the raw data. Store pointer, ZERO copying!
        void* data_ptr = (void*)ptr;
        if (dtype == 1) {
            ptr += num_elements * sizeof(int8_t);
        } else {
            ptr += num_elements * sizeof(float);
        }

        int matched = 0;
        
        // Map Embeddings
        MAP_TENSOR("embeddings.word_embeddings.weight", model->embeddings.word_embeddings);
        MAP_TENSOR("embeddings.position_embeddings.weight", model->embeddings.position_embeddings);
        MAP_TENSOR("embeddings.token_type_embeddings.weight", model->embeddings.token_type_embeddings);
        MAP_TENSOR("embeddings.LayerNorm.weight", model->embeddings.layer_norm.weight);
        MAP_TENSOR("embeddings.LayerNorm.bias", model->embeddings.layer_norm.bias);
        
        // Example mapping for layer 0 (to avoid massive file length in this step)
        MAP_TENSOR("encoder.layer.0.attention.self.query.weight", model->layers[0].attention.self.query.weight);
        MAP_TENSOR("encoder.layer.0.attention.self.query.bias", model->layers[0].attention.self.query.bias);
        
        // Map Pooler
        MAP_TENSOR("pooler.dense.weight", model->pooler.weight);
        MAP_TENSOR("pooler.dense.bias", model->pooler.bias);

        if (matched) {
            tensors_loaded++;
        }
    }
    
    printf("Successfully mapped %d tensors directly from disk.\n", tensors_loaded);
    return 0;
}
