/*
nvcc hello.cu -o cu_hello.bin
./cu_hello.bin
*/
#include <iostream>

// kernel definition
__global__ void kernel() {
    printf("Hello, CUDA!\n");
}

int main() {
    // Launch kernel
    // execution configuration is specified by <<<grid, block>>>
    kernel<<<1, 1>>>();

    // Wait for GPU to finish before printing
    cudaDeviceSynchronize();

    return 0;
}