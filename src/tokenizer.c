#include "tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

/* ------------------------------------------------------------------ */
/* Task 16 helpers                                                      */
/* ------------------------------------------------------------------ */

void tokenizer_init(Tokenizer* tok, int vocab_size) {
    tok->vocab_size    = vocab_size;
    tok->max_token_len = 0;
    tok->tokens        = (char**)calloc(vocab_size, sizeof(char*));
    tok->scores        = (float*)calloc(vocab_size, sizeof(float));
}

void tokenizer_free(Tokenizer* tok) {
    if (tok->tokens) {
        for (int i = 0; i < tok->vocab_size; ++i)
            free(tok->tokens[i]);
        free(tok->tokens);
    }
    free(tok->scores);
}

/* ------------------------------------------------------------------ */
/* Task 26: Load vocab.txt                                             */
/* Each line is one token string; its line index is its ID.            */
/* ------------------------------------------------------------------ */

int tokenizer_load_vocab(Tokenizer* tok, const char* vocab_path) {
    FILE* f = fopen(vocab_path, "r");
    if (!f) {
        fprintf(stderr, "tokenizer: cannot open %s\n", vocab_path);
        return -1;
    }

    char line[512];
    int  id = 0;

    while (id < tok->vocab_size && fgets(line, sizeof(line), f)) {
        int len = (int)strlen(line);
        while (len > 0 && (line[len-1] == '\n' || line[len-1] == '\r'))
            line[--len] = '\0';

        tok->tokens[id] = (char*)malloc(len + 1);
        memcpy(tok->tokens[id], line, len + 1);

        if (len > tok->max_token_len)
            tok->max_token_len = len;
        ++id;
    }

    fclose(f);
    tok->vocab_size = id;
    return 0;
}

/* ------------------------------------------------------------------ */
/* Internal: linear scan lookup — O(V), fine for inference path        */
/* ------------------------------------------------------------------ */

static int lookup_token(const Tokenizer* tok, const char* piece, int piece_len) {
    for (int i = 0; i < tok->vocab_size; ++i) {
        if (tok->tokens[i] &&
            (int)strlen(tok->tokens[i]) == piece_len &&
            memcmp(tok->tokens[i], piece, piece_len) == 0)
            return i;
    }
    return TOK_UNK_ID;
}

/* ------------------------------------------------------------------ */
/* Task 27: Encode — WordPiece greedy longest-match                    */
/*                                                                      */
/* Algorithm for each whitespace-delimited word:                        */
/*   1. Try to match the longest prefix in the vocabulary.             */
/*   2. The remainder is prefixed with "##" and matched again.         */
/*   3. If any sub-piece is not found, the whole word → [UNK].         */
/* ------------------------------------------------------------------ */

static int encode_word(const Tokenizer* tok,
                       const char* word, int wlen,
                       int* out_ids, int remaining) {
    int n = 0;
    int start = 0;
    int prefix = 1; /* first piece has no ##, rest have ## */

    while (start < wlen) {
        int best_end = -1;

        /* Try longest match from 'start' */
        for (int end = wlen; end > start; --end) {
            int piece_len;
            char piece[TOK_MAX_WORD + 3];

            if (!prefix) {
                piece[0] = '#'; piece[1] = '#';
                piece_len = (end - start) + 2;
                if (piece_len > TOK_MAX_WORD + 2) continue;
                memcpy(piece + 2, word + start, end - start);
                piece[piece_len] = '\0';
            } else {
                piece_len = end - start;
                if (piece_len > TOK_MAX_WORD) continue;
                memcpy(piece, word + start, piece_len);
                piece[piece_len] = '\0';
            }

            int id = lookup_token(tok, piece, piece_len);
            if (id != TOK_UNK_ID) {
                best_end = end;
                if (n >= remaining) return -1; /* output full */
                out_ids[n++] = id;
                break;
            }
        }

        if (best_end == -1) {
            /* whole word becomes [UNK] — discard any partial ids */
            if (n > 0) n = 0; /* reset word tokens */
            if (remaining < 1) return -1;
            out_ids[0] = TOK_UNK_ID;
            return 1;
        }

        start  = best_end;
        prefix = 0;
    }
    return n;
}

int tokenizer_encode(const Tokenizer* tok, const char* text,
                     int* out_ids, int max_len) {
    if (max_len < 2) return 0;

    int n = 0;
    out_ids[n++] = TOK_CLS_ID;

    const char* p = text;
    while (*p && n < max_len - 1) {
        /* skip whitespace */
        while (*p && isspace((unsigned char)*p)) ++p;
        if (!*p) break;

        /* collect one word (non-whitespace run), lowercased */
        char word[TOK_MAX_WORD];
        int  wlen = 0;
        while (*p && !isspace((unsigned char)*p) && wlen < TOK_MAX_WORD - 1) {
            word[wlen++] = (char)tolower((unsigned char)*p);
            ++p;
        }
        word[wlen] = '\0';

        int tmp_ids[TOK_MAX_WORD];
        int added = encode_word(tok, word, wlen, tmp_ids, TOK_MAX_WORD);
        if (added < 0) break;

        for (int i = 0; i < added && n < max_len - 1; ++i)
            out_ids[n++] = tmp_ids[i];
    }

    out_ids[n++] = TOK_SEP_ID;
    return n;
}

/* ------------------------------------------------------------------ */
/* Task 28: Decode — reverse lookup + strip ## continuation markers    */
/* ------------------------------------------------------------------ */

void tokenizer_decode(const Tokenizer* tok, const int* ids, int n_ids,
                      char* out_buf, int buf_len) {
    int pos = 0;
    out_buf[0] = '\0';

    for (int i = 0; i < n_ids; ++i) {
        int id = ids[i];
        if (id == TOK_CLS_ID || id == TOK_SEP_ID || id == TOK_PAD_ID)
            continue;

        const char* tok_str = (id >= 0 && id < tok->vocab_size && tok->tokens[id])
                              ? tok->tokens[id] : "[UNK]";

        int is_continuation = (tok_str[0] == '#' && tok_str[1] == '#');
        const char* piece   = is_continuation ? tok_str + 2 : tok_str;

        if (!is_continuation && pos > 0 && pos < buf_len - 1)
            out_buf[pos++] = ' ';

        int plen = (int)strlen(piece);
        int copy = (pos + plen < buf_len - 1) ? plen : (buf_len - 1 - pos);
        memcpy(out_buf + pos, piece, copy);
        pos += copy;
        out_buf[pos] = '\0';
    }
}
