#include "memory.h"
#include <string.h>

void bump_allocator_init(BumpAllocator* alloc, void* buffer, size_t capacity) {
    alloc->buffer = (uint8_t*)buffer;
    alloc->capacity = capacity;
    alloc->offset = 0;
}

void* bump_alloc(BumpAllocator* alloc, size_t size) {
    // align to 16 bytes for NEON
    size_t aligned_size = (size + 15) & ~15;
    if (alloc->offset + aligned_size > alloc->capacity) {
        return NULL; // Out of memory
    }
    void* ptr = alloc->buffer + alloc->offset;
    alloc->offset += aligned_size;
    return ptr;
}

void bump_reset(BumpAllocator* alloc) {
    alloc->offset = 0;
}

void pool_allocator_init(PoolAllocator* pool, void* buffer, size_t block_size, size_t num_blocks, uint8_t* bitmap) {
    pool->buffer = (uint8_t*)buffer;
    // Align block size to 16 bytes
    pool->block_size = (block_size + 15) & ~15;
    pool->num_blocks = num_blocks;
    pool->free_bitmap = bitmap;
    memset(pool->free_bitmap, 0, (num_blocks + 7) / 8);
}

void* pool_alloc(PoolAllocator* pool) {
    for (size_t i = 0; i < pool->num_blocks; ++i) {
        size_t byte_idx = i / 8;
        size_t bit_idx = i % 8;
        if ((pool->free_bitmap[byte_idx] & (1 << bit_idx)) == 0) {
            // Found a free block
            pool->free_bitmap[byte_idx] |= (1 << bit_idx); // Mark as used
            return pool->buffer + (i * pool->block_size);
        }
    }
    return NULL; // Out of memory
}

void pool_free(PoolAllocator* pool, void* ptr) {
    if ((uint8_t*)ptr < pool->buffer || (uint8_t*)ptr >= pool->buffer + (pool->num_blocks * pool->block_size)) {
        return; // Pointer out of bounds
    }
    size_t offset = (uint8_t*)ptr - pool->buffer;
    size_t index = offset / pool->block_size;
    size_t byte_idx = index / 8;
    size_t bit_idx = index % 8;
    pool->free_bitmap[byte_idx] &= ~(1 << bit_idx); // Mark as free
}
