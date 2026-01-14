// hello.cu
// Basic CUDA kernel demonstrating the execution model
#include <cstdio>

__global__ void hello() {
    printf("Hello from GPU! block=%d thread=%d\n",
           blockIdx.x, threadIdx.x);
}

int main() {
    hello<<<2, 3>>>();          // Launch: 2 blocks × 3 threads = 6 threads
    cudaDeviceSynchronize();    // Wait for GPU to finish
    return 0;
}
