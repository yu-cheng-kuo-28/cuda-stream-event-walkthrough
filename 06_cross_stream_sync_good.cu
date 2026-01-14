// cross_stream_sync_good.cu
// Demonstrates proper cross-stream synchronization using events
// Consumer waits for producer via cudaStreamWaitEvent
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
        
        cudaEvent_t dataReady;
        cudaEventCreate(&dataReady);

        // Launch producer
        producer<<<1, 1, 0, streamA>>>(d_data);

        // Record event when producer finishes
        cudaEventRecord(dataReady, streamA);
        
        // Stream B: wait for signal, then consume
        cudaStreamWaitEvent(streamB, dataReady);
        
        // Launch consumers - now they wait for producer
        for (int i = 0; i < 5; ++i) {
            consumer<<<1, 1, 0, streamB>>>(d_data);
        }
        
        cudaDeviceSynchronize();
        
        cudaEventDestroy(dataReady);
        cudaStreamDestroy(streamB);
        cudaStreamDestroy(streamA);
        cudaFree(d_data);
    }
    
    return 0;
}
