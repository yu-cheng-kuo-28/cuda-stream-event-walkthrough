// cross_stream_sync_bad.cu
// Demonstrates race condition WITHOUT proper synchronization
// Consumer may read before producer writes
#include <cstdio>

__global__ void producer(int *data) {
    // Make producer REALLY slow
    for (volatile int i = 0; i < 100000000; ++i);  // 100M iterations
    *data = 42;
    printf("Producer: wrote %d\n", *data);
}

__global__ void consumer(int *data) {
    printf("Consumer: read %d\n", *data);
}

int main() {
    for (int trial = 0; trial < 10; ++trial) {
        printf("\n=== Trial %d ===\n", trial);
        
        int *d_data;
        cudaMalloc(&d_data, sizeof(int));
        
        int initValue = 10;
        cudaMemcpy(d_data, &initValue, sizeof(int), cudaMemcpyHostToDevice);
        
        cudaStream_t streamA, streamB;
        cudaStreamCreate(&streamA);
        cudaStreamCreate(&streamB);
        
        // Launch producer
        producer<<<1, 1, 0, streamA>>>(d_data);
        
        // Launch consumers IMMEDIATELY in rapid succession
        // WITHOUT waiting for producer - RACE CONDITION!
        for (int i = 0; i < 5; ++i) {
            consumer<<<1, 1, 0, streamB>>>(d_data);
        }
        
        cudaDeviceSynchronize();
        
        cudaStreamDestroy(streamA);
        cudaStreamDestroy(streamB);
        cudaFree(d_data);
    }
    
    return 0;
}
