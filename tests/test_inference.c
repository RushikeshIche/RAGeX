#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "../include/model.h"
#include "../include/loader.h"
#include "../include/memory.h"
#include "../include/mmap_utils.h"
#include "../include/inference.h"

#define BUMP_SIZE (64 * 1024 * 1024)
#define COSINE_PASS_THRESHOLD 0.99f

static float cosine_similarity(const float* a, const float* b, int n) {
    float dot = 0.0f, na = 0.0f, nb = 0.0f;
    for (int i = 0; i < n; ++i) {
        dot += a[i] * b[i];
        na  += a[i] * a[i];
        nb  += b[i] * b[i];
    }
    return dot / (sqrtf(na) * sqrtf(nb) + 1e-9f);
}

static int load_ref_tokens(const char* path, int* ids, int* seq_len) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); return -1; }
    fread(seq_len, sizeof(int), 1, f);
    fread(ids, sizeof(int), *seq_len, f);
    fclose(f);
    return 0;
}

static int load_ref_embedding(const char* path, float* emb, int* dim) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); return -1; }
    fread(dim, sizeof(int), 1, f);
    fread(emb, sizeof(float), *dim, f);
    fclose(f);
    return 0;
}

int main(void) {
    MiniLM   model;
    MmapFile mf;

    if (load_minilm_model("models/minilm_weights.bin", &model, &mf) != 0)
        return 1;

    uint8_t* bump_buf = (uint8_t*)malloc(BUMP_SIZE);
    BumpAllocator mem;
    bump_allocator_init(&mem, bump_buf, BUMP_SIZE);

    int token_ids[512];
    int seq_len = 0;
    if (load_ref_tokens("data/ref_tokens.bin", token_ids, &seq_len) != 0)
        return 1;

    float ref_emb[HIDDEN_SIZE];
    int   ref_dim = 0;
    if (load_ref_embedding("data/ref_embedding.bin", ref_emb, &ref_dim) != 0)
        return 1;

    float c_emb[HIDDEN_SIZE];
    forward_encoder(&model, token_ids, seq_len, c_emb, &mem);

    float sim = cosine_similarity(c_emb, ref_emb, HIDDEN_SIZE);
    printf("Cosine similarity vs PyTorch: %.6f\n", sim);
    printf("C   first 4: %.5f %.5f %.5f %.5f\n", c_emb[0],   c_emb[1],   c_emb[2],   c_emb[3]);
    printf("Ref first 4: %.5f %.5f %.5f %.5f\n", ref_emb[0], ref_emb[1], ref_emb[2], ref_emb[3]);

    int pass = sim >= COSINE_PASS_THRESHOLD;
    printf("Validation: %s (threshold=%.2f)\n", pass ? "PASS" : "FAIL", COSINE_PASS_THRESHOLD);

    free(bump_buf);
    munmap_file(&mf);
    return pass ? 0 : 1;
}
