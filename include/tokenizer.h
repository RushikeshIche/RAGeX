#ifndef RAGEX_TOKENIZER_H
#define RAGEX_TOKENIZER_H

#include <stdint.h>
#include <stddef.h>

/**
 * Basic Tokenizer structure.
 * This holds the vocabulary needed to convert text strings into integer IDs
 * and integer IDs back into text strings.
 */
typedef struct {
    char** tokens;      // Array of token strings, where index == token ID
    float* scores;      // Optional token scores (used in some BPE implementations)
    int vocab_size;     // Total number of tokens in the vocabulary
    int max_token_len;  // Length of the longest token in the vocabulary
} Tokenizer;

void tokenizer_init(Tokenizer* tokenizer, int vocab_size);

void tokenizer_free(Tokenizer* tokenizer);

#endif
