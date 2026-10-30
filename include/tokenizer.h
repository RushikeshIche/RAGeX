#ifndef RAGEX_TOKENIZER_H
#define RAGEX_TOKENIZER_H

#include <stdint.h>
#include <stddef.h>

#define TOK_UNK_ID   100
#define TOK_CLS_ID   101
#define TOK_SEP_ID   102
#define TOK_PAD_ID     0
#define TOK_MAX_WORD 128

typedef struct {
    char**  tokens;
    float*  scores;
    int     vocab_size;
    int     max_token_len;
} Tokenizer;

/* Task 16 */
void tokenizer_init(Tokenizer* tok, int vocab_size);
void tokenizer_free(Tokenizer* tok);

/* Task 26: load vocab.txt (one token per line, index == id) */
int  tokenizer_load_vocab(Tokenizer* tok, const char* vocab_path);

/* Task 27: encode a UTF-8 string into token IDs (WordPiece / BPE).
   Returns number of tokens written into out_ids.
   Prepends [CLS]=101, appends [SEP]=102.
   max_len is the maximum number of ids (including CLS/SEP). */
int  tokenizer_encode(const Tokenizer* tok, const char* text,
                      int* out_ids, int max_len);

/* Task 28: decode token ID array back into a human-readable string.
   Writes into out_buf (null-terminated). Strips ## continuation markers. */
void tokenizer_decode(const Tokenizer* tok, const int* ids, int n_ids,
                      char* out_buf, int buf_len);

#endif
