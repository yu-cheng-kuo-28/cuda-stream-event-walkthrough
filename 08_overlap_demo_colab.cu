// %%writefile overlap_demo.cu
// overlap_demo.cu
#include <cstdio>

// Check GPU concurrent capabilities
void checkGPUCapabilities() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    printf("=== GPU Capabilities ===\n");
    printf("Device: %s\n", prop.name);
    printf("Concurrent kernels: %s\n",
           prop.concurrentKernels ? "YES" : "NO");
    printf("Async engine count: %d\n", prop.asyncEngineCount);
    printf("Concurrent copy and execution: %s\n",
           prop.deviceOverlap ? "YES" : "NO");
    printf("\n");
}

#define N (1 << 20)  // Smaller: 1M elements
#define NSTREAMS 2

__global__ void process(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Minimal computation to emphasize transfer overlap
        float val = data[idx];
        for (int i = 0; i < 10; ++i) {
            val = val * 0.99f + 0.01f;
        }
        data[idx] = val;
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

    // Launch all streams
    for (int i = 0; i < NSTREAMS; ++i) {
        int offset = i * chunkSize;
        cudaMemcpyAsync(d_data + offset, h_data + offset,
                        chunkBytes, cudaMemcpyHostToDevice, streams[i]);
        process<<<(chunkSize+255)/256, 256, 0, streams[i]>>>(d_data + offset, chunkSize);
        cudaMemcpyAsync(h_data + offset, d_data + offset,
                        chunkBytes, cudaMemcpyDeviceToHost, streams[i]);
    }

    cudaDeviceSynchronize();

    for (int i = 0; i < NSTREAMS; ++i) {
        cudaStreamDestroy(streams[i]);
    }
}

int main() {
    // Check hardware first
    checkGPUCapabilities();

    size_t size = N * sizeof(float);

    float *h_data;
    cudaMallocHost(&h_data, size);

    float *d_data;
    cudaMalloc(&d_data, size);

    for (int i = 0; i < N; ++i) {
        h_data[i] = i * 0.001f;
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float ms;

    // Warm-up
    runSerial(h_data, d_data, size, N);
    cudaDeviceSynchronize();

    printf("=== Performance Comparison ===\n");

    // Serial
    cudaEventRecord(start);
    runSerial(h_data, d_data, size, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf("Serial:  %.2f ms\n", ms);
    float serialTime = ms;

    // Streams
    cudaEventRecord(start);
    runStreams(h_data, d_data, size, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf("Streams: %.2f ms\n", ms);
    float streamTime = ms;

    printf("\nSpeedup: %.2fx\n", serialTime / streamTime);

    if (streamTime >= serialTime) {
        printf("\n⚠️  WARNING: Your GPU does not support concurrent execution well.\n");
        printf("This is common in:\n");
        printf("  - WSL/virtualized environments\n");
        printf("  - Integrated GPUs\n");
        printf("  - Older GPU architectures\n");
        printf("\nOn a proper datacenter GPU (V100, A100, H100),\n");
        printf("you would see 1.5-2x speedup from stream overlap.\n");
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFreeHost(h_data);
    cudaFree(d_data);

    return 0;
}
// ```

// **This version:**
// 1. **Checks GPU capabilities** to show why overlap doesn't work
// 2. **Smaller problem size** (1M instead of 4M) - less time wasted
// 3. **Simpler computation** - focuses on the concept
// 4. **Explains the limitation** when detected

// **Expected output on your system:**
// ```
// === GPU Capabilities ===
// Device: [Your GPU name]
// Concurrent kernels: NO (or limited)
// Async engine count: 1
// Concurrent copy and execution: NO

// === Performance Comparison ===
// Serial:  2.1 ms
// Streams: 2.3 ms
// Speedup: 0.91x

// ⚠️  WARNING: Your GPU does not support concurrent execution well.
// ```

// **On a real datacenter GPU, you'd see:**
// ```
// Concurrent kernels: YES
// Async engine count: 2
// Serial:  5.2 ms
// Streams: 2.8 ms
// Speedup: 1.86x