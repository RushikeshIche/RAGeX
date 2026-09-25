#ifndef RAGEX_MEMORY_H
#define RAGEX_MEMORY_H

#include <stddef.h>
#include <stdint.h>

// Bump allocator for fast temporary allocations
typedef struct {
    uint8_t* buffer;
    size_t capacity;
    size_t offset;
} BumpAllocator;

void bump_allocator_init(BumpAllocator* alloc, void* buffer, size_t capacity);
void* bump_alloc(BumpAllocator* alloc, size_t size);
void bump_reset(BumpAllocator* alloc);

// Pool allocator for fixed size tensors
typedef struct {
    uint8_t* buffer;
    size_t block_size;
    size_t num_blocks;
    uint8_t* free_bitmap;
} PoolAllocator;

void pool_allocator_init(PoolAllocator* pool, void* buffer, size_t block_size, size_t num_blocks, uint8_t* bitmap);
void* pool_alloc(PoolAllocator* pool);
void pool_free(PoolAllocator* pool, void* ptr);

#endif 
