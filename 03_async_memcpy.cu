// async_memcpy.cu
// Demonstrating async memory transfers with streams
#include <cstdio>

__global__ void process(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] *= 2.0f;
    }
}

int main() {
    const int N = 1 << 20;  // 1M elements
    const size_t size = N * sizeof(float);
    
    // Pinned host memory (required for async)
    float *h_data;
    cudaMallocHost(&h_data, size);
    
    // Device memory
    float *d_data;
    cudaMalloc(&d_data, size);
    
    // Initialize
    for (int i = 0; i < N; ++i) {
        h_data[i] = i;
    }
    
    // Create stream
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    // Async operations in stream
    cudaMemcpyAsync(d_data, h_data, size, cudaMemcpyHostToDevice, stream);
    process<<<(N+255)/256, 256, 0, stream>>>(d_data, N);
    cudaMemcpyAsync(h_data, d_data, size, cudaMemcpyDeviceToHost, stream);
    
    // Wait for stream to complete
    cudaStreamSynchronize(stream);
    
    // Verify
    printf("h_data[100] = %f (expected 200.0)\n", h_data[100]);
    
    // Cleanup
    cudaStreamDestroy(stream);
    cudaFreeHost(h_data);
    cudaFree(d_data);
    
    return 0;
}
