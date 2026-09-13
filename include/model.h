#ifndef RAGEX_MODEL_H
#define RAGEX_MODEL_H

#include "types.h"

// Constants for MiniLM-L6-v2 architecture
#define VOCAB_SIZE 30522
#define MAX_SEQ_LEN 512
#define HIDDEN_SIZE 384
#define INTERMEDIATE_SIZE 1536
#define NUM_ATTENTION_HEADS 12
#define NUM_HIDDEN_LAYERS 6

// Basic building blocks
typedef struct {
    Tensor weight;
    Tensor bias; // Some layers might not have bias, we'll keep data=NULL if so
} LinearLayer;

typedef struct {
    Tensor weight;
    Tensor bias;
} LayerNorm;

// Embedding Layer
typedef struct {
    Tensor word_embeddings;
    Tensor position_embeddings;
    Tensor token_type_embeddings;
    LayerNorm layer_norm;
} Embeddings;

// Self Attention block
typedef struct {
    LinearLayer query;
    LinearLayer key;
    LinearLayer value;
} SelfAttention;

typedef struct {
    LinearLayer dense;
    LayerNorm layer_norm;
} AttentionOutput;

typedef struct {
    SelfAttention self;
    AttentionOutput output;
} Attention;

// Feed Forward block
typedef struct {
    LinearLayer dense;
} Intermediate;

typedef struct {
    LinearLayer dense;
    LayerNorm layer_norm;
} OutputBlock;

// A single Transformer Layer (Encoder)
typedef struct {
    Attention attention;
    Intermediate intermediate;
    OutputBlock output;
} TransformerLayer;

// The complete MiniLM model
typedef struct {
    Embeddings embeddings;
    TransformerLayer layers[NUM_HIDDEN_LAYERS];
    LinearLayer pooler;
} MiniLM;

#endif // RAGEX_MODEL_H
