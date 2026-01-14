// cuda_check.h
// Error checking macro for CUDA API calls
#ifndef CUDA_CHECK_H
#define CUDA_CHECK_H

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

#endif // CUDA_CHECK_H
