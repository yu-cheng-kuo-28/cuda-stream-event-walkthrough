// streams_basic.cu
// Basic stream creation, usage, and destruction
#include <cstdio>

__global__ void kernel(int id) {
    // Simulate work
    for (volatile int i = 0; i < 1000000; ++i);
    printf("Kernel %d complete\n", id);
}

int main() {
    cudaStream_t stream1, stream2;
    
    // Create streams
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    
    // Launch kernels in different streams
    kernel<<<1, 1, 0, stream1>>>(1);  // 4th param = stream
    kernel<<<1, 1, 0, stream2>>>(2);
    
    // Wait for both streams
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);
    
    // Cleanup
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    
    return 0;
}
