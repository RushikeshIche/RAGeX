#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "../include/math.h"
#include "../include/memory.h"

void bench_gemm() {
    int M = 256, N = 256, K = 256; 
    float* A = (float*)malloc(M * K * sizeof(float));
    float* B = (float*)malloc(K * N * sizeof(float));
    float* C = (float*)malloc(M * N * sizeof(float));

    if (!A || !B || !C) {
        printf("Failed to allocate memory\n");
        return;
    }

    // Initialize with dummy data
    for (int i = 0; i < M * K; ++i) A[i] = 1.0f;
    for (int i = 0; i < K * N; ++i) B[i] = 1.0f;
    for (int i = 0; i < M * N; ++i) C[i] = 0.0f;

    clock_t start, end;
    int iterations = 100;

    // Warm up
    gemm_f32_baseline(M, N, K, 1.0f, A, B, 0.0f, C);
    
    printf("Running GEMM benchmark (M=%d, N=%d, K=%d) over %d iterations...\n", M, N, K, iterations);

    // Baseline Benchmark
    start = clock();
    for (int i = 0; i < iterations; ++i) {
        gemm_f32_baseline(M, N, K, 1.0f, A, B, 0.0f, C);
    }
    end = clock();
    double baseline_ms = ((double)(end - start) / CLOCKS_PER_SEC) * 1000.0 / iterations;
    printf("Baseline GEMM: %.3f ms per iteration\n", baseline_ms);

#ifdef __ARM_NEON
    // NEON Benchmark
    start = clock();
    for (int i = 0; i < iterations; ++i) {
        gemm_f32_neon(M, N, K, 1.0f, A, B, 0.0f, C);
    }
    end = clock();
    double neon_ms = ((double)(end - start) / CLOCKS_PER_SEC) * 1000.0 / iterations;
    printf("NEON GEMM:     %.3f ms per iteration\n", neon_ms);
    printf("Speedup:       %.2fx\n", baseline_ms / neon_ms);
#else
    printf("NEON GEMM:     [Skipped - Not compiled with __ARM_NEON]\n");
#endif

    free(A);
    free(B);
    free(C);
}

void bench_memory_allocators() {
    printf("\nRunning memory allocator benchmark...\n");
    
    size_t pool_size = 1024 * 1024; // 1 MB
    uint8_t* buffer = (uint8_t*)malloc(pool_size);
    BumpAllocator bump;
    
    clock_t start = clock();
    bump_allocator_init(&bump, buffer, pool_size);
    for(int i = 0; i < 10000; i++) {
        bump_alloc(&bump, 32);
    }
    clock_t end = clock();
    double ms = ((double)(end - start) / CLOCKS_PER_SEC) * 1000.0;
    
    printf("Bump allocator (10k allocs): %.3f ms\n", ms);
    free(buffer);
}

int main() {
    printf("=== RAGeX Benchmarks ===\n\n");
    bench_gemm();
    bench_memory_allocators();
    return 0;
}
