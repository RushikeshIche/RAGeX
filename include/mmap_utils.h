#ifndef RAGEX_MMAP_UTILS_H
#define RAGEX_MMAP_UTILS_H

#include <stddef.h>

typedef struct {
    void* data;
    size_t size;
#ifdef _WIN32
    void* hFile;
    void* hMapping;
#else
    int fd;
#endif
} MmapFile;

// Maps a file into memory (read-only)
// Returns 0 on success, -1 on failure
int mmap_file_read(const char* filepath, MmapFile* mf);

// Unmaps the file and cleans up handles
void munmap_file(MmapFile* mf);

#endif 
