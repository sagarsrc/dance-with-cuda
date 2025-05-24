/**
 * Basic CUDA Hello World program
 *
 * CUDA Terminology:
 * - Host: CPU and its memory
 * - Device: GPU and its memory
 * - Kernel: CUDA C function that runs on the device
 * - Grid: Collection of thread blocks
 * - Block: Collection of threads
 * - Thread: Smallest unit of parallel execution
 * - Warp: How many threads are executed in parallel on the GPU
 *
 * Grouping of threads is defined by programmer
 *
 * Compilation: nvcc hello.cu -o cu_hello.bin
 * Execution: ./cu_hello.bin
 */

#include <iostream>

/**
 * Simple kernel that prints "Hello, CUDA!"
 *
 * The __global__ keyword indicates this is a kernel function that:
 * - Can be called from host code (CPU)
 * - Executes on the device (GPU)
 * - Must return void
 */
__global__ void kernel()
{
    printf("Hello, CUDA!\n");
}

int main()
{
    // Launch kernel with 1 thread block containing 1 thread
    // The <<<...>>> syntax specifies the kernel's execution configuration:
    // - First parameter: Number of blocks in the grid
    // - Second parameter: Number of threads per block
    kernel<<<1, 1>>>();

    // Wait for GPU to finish before exiting
    cudaDeviceSynchronize();

    return 0;
}