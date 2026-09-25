#ifndef RAGEX_INFERENCE_H
#define RAGEX_INFERENCE_H

#include "model.h"

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

#endif 
