// event_timing.cu
// Using CUDA events to measure kernel execution time
#include <cstdio>

__global__ void heavyKernel() {
    for (volatile int i = 0; i < 100000000; ++i);
}

int main() {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Record start
    cudaEventRecord(start);
    
    // Work to measure
    heavyKernel<<<1, 1>>>();
    
    // Record stop
    cudaEventRecord(stop);
    
    // Wait for stop event
    cudaEventSynchronize(stop);
    
    // Calculate elapsed time
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    printf("Kernel time: %.3f ms\n", ms);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}
