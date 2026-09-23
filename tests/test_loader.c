#include <stdio.h>
#include <stdlib.h>
#include "../include/model.h"
#include "../include/loader.h"

int main() {
    MiniLM model = {0};
    MmapFile mf = {0};

    printf("=== RAGeX Model Loader Test ===\n");
    
    // Attempt to load the model exported by the Python script
    if (load_minilm_model("models/minilm_weights.bin", &model, &mf) != 0) {
        printf("FAIL: Failed to load model. Did you run the python export script?\n");
        return 1;
    }

    int passed = 1;

    // 1. Verify standard FP32 layer mapping
    if (model.embeddings.word_embeddings.data == NULL) {
        printf("FAIL: Word embeddings pointer is NULL\n");
        passed = 0;
    }
    if (model.embeddings.word_embeddings.shape[0] != VOCAB_SIZE || 
        model.embeddings.word_embeddings.shape[1] != HIDDEN_SIZE) {
        printf("FAIL: Word embeddings shape mismatch (%dx%d instead of %dx%d)\n", 
               model.embeddings.word_embeddings.shape[0], model.embeddings.word_embeddings.shape[1],
               VOCAB_SIZE, HIDDEN_SIZE);
        passed = 0;
    }
    if (model.embeddings.word_embeddings.dtype != DTYPE_F32) {
        printf("FAIL: Word embeddings should be F32\n");
        passed = 0;
    }

    // 2. Verify INT8 Quantized layer mapping
    if (model.layers[0].attention.self.query.weight.data == NULL) {
        printf("FAIL: Layer 0 Query weight pointer is NULL\n");
        passed = 0;
    }
    if (model.layers[0].attention.self.query.weight.dtype != DTYPE_INT8) {
        printf("FAIL: Layer 0 Query weight should be INT8 quantized\n");
        passed = 0;
    }
    if (model.layers[0].attention.self.query.weight.scale <= 0.0f) {
        printf("FAIL: Layer 0 Query weight scale factor is missing/invalid\n");
        passed = 0;
    }

    // 3. Verify Pooler layer mapping
    if (model.pooler.weight.data == NULL) {
        printf("FAIL: Pooler dense weight is NULL\n");
        passed = 0;
    }

    if (passed) {
        printf("\nSUCCESS: C Model Loader memory-mapped the binary correctly!\n");
        printf(" - Structs aligned properly.\n");
        printf(" - INT8 quantization scale factors successfully parsed.\n");
    }

    munmap_file(&mf);
    return passed ? 0 : 1;
}
