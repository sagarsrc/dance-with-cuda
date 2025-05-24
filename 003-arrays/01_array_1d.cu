/**
 * Enhanced 1D array memory organization visualization
 * Remember array is always stored in contiguous memory
 * Think of array as row of houses with mailboxes
 * Which mailman visits which mailbox first can vary on each day

 * Similarly array is stored in contiguous memory
 * Which thread accesses which memory location first can vary on each run
 *
 */

#include <cuda_runtime.h>
#include <bits/stdc++.h>

using namespace std;

__global__ void printArrayWithMemoryInfo(int *a, int N)
{
    /**
     * Print array values along with memory addresses to show memory layout
     */
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (tid < N)
    {
        printf("thread %d: a[%d] = %d, memory address = %p, offset = %lu bytes\n",
               tid, tid, a[tid], &a[tid], tid * sizeof(int));
    }
}

__global__ void printMemoryPattern(int *a, int N)
{
    /**
     * Show memory access pattern and addresses
     */
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (tid < N)
    {
        printf("Thread %d accesses memory at %p (element a[%d] = %d)\n",
               tid, &a[tid], tid, a[tid]);
    }
}

void printHostMemoryInfo(vector<int>& h_a)
{
    printf("\n=== HOST MEMORY LAYOUT ===\n");
    printf("Host array base address: %p\n", h_a.data());
    printf("Size of int: %d bytes\n", (int)sizeof(int));
    printf("Array size: %d elements = %d bytes\n", (int)h_a.size(), (int)(h_a.size() * sizeof(int)));

    for (int i = 0; i < (int)h_a.size(); i++)
    {
        printf("h_a[%d] = %d, address = %p, offset = %d bytes\n",
               i, h_a[i], &h_a[i], i * (int)sizeof(int));
    }
    printf("\n");
}

void printMemoryAlignment(int *d_a, int N)
{
    printf("=== DEVICE MEMORY INFO ===\n");
    printf("Device array base address: %p\n", d_a);
    printf("Array spans from %p to %p\n", d_a, d_a + N - 1);
    printf("Total memory allocated: %d bytes\n", N * (int)sizeof(int));
    printf("Memory alignment: addresses should be 4-byte aligned for int\n\n");
}

int main(int argc, char **argv)
{
    int N = 12; // Using 12 elements to show multiple warps
    vector<int> h_a(N);
    int *d_a;

    // threads per block
    int THREADS = 8; // Smaller block size to see memory pattern clearly
    cout << "threads per block: " << THREADS << endl;
    // blocks per grid
    int BLOCKS = (N + THREADS - 1) / THREADS;
    cout << "blocks per grid: " << BLOCKS << endl;

    // Initialize array with values
    for (int i = 0; i < N; i++)
    {
        h_a[i] = i * 10; // Using multiples of 10 for clarity
    }

    // Print host memory information
    printHostMemoryInfo(h_a);

    // Allocate and copy to device
    cudaMalloc(&d_a, N * sizeof(int));
    cudaMemcpy(d_a, h_a.data(), N * sizeof(int), cudaMemcpyHostToDevice);

    // Print device memory information
    printMemoryAlignment(d_a, N);

    printf("=== DEVICE MEMORY ACCESS PATTERNS ===\n");
    printf("Grid configuration: %d blocks × %d threads = %d total threads\n", BLOCKS, THREADS, BLOCKS * THREADS);
    printf("Array size: %d elements\n\n", N);

    printf("1. Array values with memory addresses:\n");
    printArrayWithMemoryInfo<<<BLOCKS, THREADS>>>(d_a, N);
    cudaDeviceSynchronize();

    printf("\n2. Memory access pattern:\n");
    printMemoryPattern<<<BLOCKS, THREADS>>>(d_a, N);
    cudaDeviceSynchronize();


    cudaFree(d_a);
    return 0;
}


/*
Output:

threads per block: 8
blocks per grid: 2

=== HOST MEMORY LAYOUT ===
Host array base address: 0x6142634535f0
Size of int: 4 bytes
Array size: 12 elements = 48 bytes
h_a[0] = 0, address = 0x6142634535f0, offset = 0 bytes
h_a[1] = 10, address = 0x6142634535f4, offset = 4 bytes
h_a[2] = 20, address = 0x6142634535f8, offset = 8 bytes
h_a[3] = 30, address = 0x6142634535fc, offset = 12 bytes
h_a[4] = 40, address = 0x614263453600, offset = 16 bytes
h_a[5] = 50, address = 0x614263453604, offset = 20 bytes
h_a[6] = 60, address = 0x614263453608, offset = 24 bytes
h_a[7] = 70, address = 0x61426345360c, offset = 28 bytes
h_a[8] = 80, address = 0x614263453610, offset = 32 bytes
h_a[9] = 90, address = 0x614263453614, offset = 36 bytes
h_a[10] = 100, address = 0x614263453618, offset = 40 bytes
h_a[11] = 110, address = 0x61426345361c, offset = 44 bytes

=== DEVICE MEMORY INFO ===
Device array base address: 0x7230d7200000
Array spans from 0x7230d7200000 to 0x7230d720002c
Total memory allocated: 48 bytes
Memory alignment: addresses should be 4-byte aligned for int

=== DEVICE MEMORY ACCESS PATTERNS ===
Grid configuration: 2 blocks × 8 threads = 16 total threads
Array size: 12 elements

1. Array values with memory addresses:
thread 8: a[8] = 80, memory address = 0x7230d7200020, offset = 32 bytes
thread 9: a[9] = 90, memory address = 0x7230d7200024, offset = 36 bytes
thread 10: a[10] = 100, memory address = 0x7230d7200028, offset = 40 bytes
thread 11: a[11] = 110, memory address = 0x7230d720002c, offset = 44 bytes
thread 0: a[0] = 0, memory address = 0x7230d7200000, offset = 0 bytes
thread 1: a[1] = 10, memory address = 0x7230d7200004, offset = 4 bytes
thread 2: a[2] = 20, memory address = 0x7230d7200008, offset = 8 bytes
thread 3: a[3] = 30, memory address = 0x7230d720000c, offset = 12 bytes
thread 4: a[4] = 40, memory address = 0x7230d7200010, offset = 16 bytes
thread 5: a[5] = 50, memory address = 0x7230d7200014, offset = 20 bytes
thread 6: a[6] = 60, memory address = 0x7230d7200018, offset = 24 bytes
thread 7: a[7] = 70, memory address = 0x7230d720001c, offset = 28 bytes

2. Memory access pattern:
Thread 8 accesses memory at 0x7230d7200020 (element a[8] = 80)
Thread 9 accesses memory at 0x7230d7200024 (element a[9] = 90)
Thread 10 accesses memory at 0x7230d7200028 (element a[10] = 100)
Thread 11 accesses memory at 0x7230d720002c (element a[11] = 110)
Thread 0 accesses memory at 0x7230d7200000 (element a[0] = 0)
Thread 1 accesses memory at 0x7230d7200004 (element a[1] = 10)
Thread 2 accesses memory at 0x7230d7200008 (element a[2] = 20)
Thread 3 accesses memory at 0x7230d720000c (element a[3] = 30)
Thread 4 accesses memory at 0x7230d7200010 (element a[4] = 40)
Thread 5 accesses memory at 0x7230d7200014 (element a[5] = 50)
Thread 6 accesses memory at 0x7230d7200018 (element a[6] = 60)
Thread 7 accesses memory at 0x7230d720001c (element a[7] = 70)

*/