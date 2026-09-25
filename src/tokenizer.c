#include "tokenizer.h"
#include <stdlib.h>
#include <string.h>

void tokenizer_init(Tokenizer* tokenizer, int vocab_size) {
    tokenizer->vocab_size = vocab_size;
    tokenizer->max_token_len = 0;
    
    // Allocate arrays for tokens and scores
    tokenizer->tokens = (char**)malloc(vocab_size * sizeof(char*));
    tokenizer->scores = (float*)malloc(vocab_size * sizeof(float));
    
    // Initialize to NULL / 0.0
    for (int i = 0; i < vocab_size; ++i) {
        tokenizer->tokens[i] = NULL;
        tokenizer->scores[i] = 0.0f;
    }
}

void tokenizer_free(Tokenizer* tokenizer) {
    if (tokenizer->tokens) {
        for (int i = 0; i < tokenizer->vocab_size; ++i) {
            if (tokenizer->tokens[i]) {
                free(tokenizer->tokens[i]); // Free individual strings
            }
        }
        free(tokenizer->tokens); // Free the array of pointers
    }
    
    if (tokenizer->scores) {
        free(tokenizer->scores);
    }
}
