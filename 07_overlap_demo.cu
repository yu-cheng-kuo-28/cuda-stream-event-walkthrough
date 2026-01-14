// overlap_demo.cu
// Complete demonstration of overlapping transfers and compute
// Measures actual speedup from using multiple streams
#include <cstdio>

#define N (1 << 22)  // 4M elements
#define NSTREAMS 4

__global__ void process(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Simulate heavy computation
        for (int i = 0; i < 100; ++i) {
            data[idx] = sinf(data[idx]) * cosf(data[idx]);
        }
    }
}

void runSerial(float *h_data, float *d_data, size_t size, int n) {
    cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);
    process<<<(n+255)/256, 256>>>(d_data, n);
    cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost);
}

void runStreams(float *h_data, float *d_data, size_t size, int n) {
    cudaStream_t streams[NSTREAMS];
    for (int i = 0; i < NSTREAMS; ++i) {
        cudaStreamCreate(&streams[i]);
    }
    
    int chunkSize = n / NSTREAMS;
    size_t chunkBytes = chunkSize * sizeof(float);
    
    for (int i = 0; i < NSTREAMS; ++i) {
        int offset = i * chunkSize;
        cudaMemcpyAsync(d_data + offset, h_data + offset, 
                        chunkBytes, cudaMemcpyHostToDevice, streams[i]);
        process<<<(chunkSize+255)/256, 256, 0, streams[i]>>>(d_data + offset, chunkSize);
        cudaMemcpyAsync(h_data + offset, d_data + offset,
                        chunkBytes, cudaMemcpyDeviceToHost, streams[i]);
    }
    
    for (int i = 0; i < NSTREAMS; ++i) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }
}

int main() {
    size_t size = N * sizeof(float);
    
    // Pinned host memory
    float *h_data;
    cudaMallocHost(&h_data, size);
    
    float *d_data;
    cudaMalloc(&d_data, size);
    
    // Initialize
    for (int i = 0; i < N; ++i) h_data[i] = i * 0.001f;
    
    // Timing events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float ms;
    
    // Warm-up
    runSerial(h_data, d_data, size, N);
    cudaDeviceSynchronize();
    
    // Time serial
    cudaEventRecord(start);
    runSerial(h_data, d_data, size, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf("Serial:  %.2f ms\n", ms);
    
    // Time streams
    cudaEventRecord(start);
    runStreams(h_data, d_data, size, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf("Streams: %.2f ms\n", ms);
    
    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFreeHost(h_data);
    cudaFree(d_data);
    
    return 0;
}
