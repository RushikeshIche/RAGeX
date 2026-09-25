#ifndef RAGEX_LOADER_H
#define RAGEX_LOADER_H

#include "model.h"
#include "mmap_utils.h"

// Maps the binary file into memory and wires up the pointers in the MiniLM struct
// Returns 0 on success, -1 on failure.
int load_minilm_model(const char* filepath, MiniLM* model, MmapFile* mf);

#endif
