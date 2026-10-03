#ifndef RAGEX_INFERENCE_H
#define RAGEX_INFERENCE_H

#include "model.h"
#include "memory.h"

/**
 * Performs the forward pass for the Word Embedding layer.
 * Takes an array of integer token IDs and outputs their dense vector representations.
 * 
 * @param model The loaded MiniLM model
 * @param token_ids Array of input token IDs
 * @param seq_len Number of tokens in the sequence
 * @param output Pre-allocated buffer of size (seq_len * HIDDEN_SIZE) floats
 */
void forward_word_embeddings(const MiniLM* model, const int* token_ids, int seq_len, float* output);

/**
 * Adds absolute positional embeddings and token type embeddings to the word vectors.
 * 
 * @param model The loaded MiniLM model
 * @param seq_len Number of tokens in the sequence
 * @param output The buffer containing the word embeddings (will be modified in-place)
 */
void add_positional_embeddings(const MiniLM* model, int seq_len, float* output);

/**
 * Executes the Multi-Head Self-Attention forward pass.
 * 
 * @param attn The Attention structures (Query, Key, Value weights)
 * @param seq_len The length of the sequence
 * @param x The input sequence vectors (size: seq_len * HIDDEN_SIZE)
 * @param output The output buffer to store the results
 * @param mem A BumpAllocator to provide fast temporary memory for Q, K, and V arrays
 */
void forward_attention(const Attention* attn, int seq_len, const float* x, float* output, BumpAllocator* mem);

/**
 * Executes the Feed-Forward Network (FFN / MLP) forward pass.
 * Each token's vector is expanded from 384 -> 1536 dimensions (with GELU activation),
 * then compressed back down 1536 -> 384.
 *
 * @param intermediate The first linear layer (expand: 384 -> 1536)
 * @param output_block The second linear layer (compress: 1536 -> 384)
 * @param seq_len Number of tokens
 * @param x Input buffer (seq_len * HIDDEN_SIZE floats)
 * @param out Output buffer (seq_len * HIDDEN_SIZE floats)
 * @param mem BumpAllocator for temporary intermediate buffer
 */
void forward_ffn(const Intermediate* intermediate, const OutputBlock* output_block,
                 int seq_len, const float* x, float* out, BumpAllocator* mem);

/**
 * Applies Layer Normalization to a buffer of token vectors (in-place).
 * Normalizes each token's 384-dim vector to have mean=0, std=1, then
 * applies learned scale (weight) and shift (bias) parameters.
 *
 * @param ln The LayerNorm struct containing weight and bias tensors
 * @param x  Buffer of (seq_len * HIDDEN_SIZE) floats, modified in-place
 * @param seq_len Number of tokens
 */
void layer_norm(const LayerNorm* ln, float* x, int seq_len);

#endif
