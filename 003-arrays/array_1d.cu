/**
 * simple 1D array and printing
 */

#include <cuda_runtime.h>
#include <bits/stdc++.h>

using namespace std;

__global__ void printArrayWithoutTidLimit(int *a, int N)
{
    /**
     * Here we do not use any if condition on the thread id.
     * It results in threads accessing memory locations of *a
     * for which there is no data you will see 0s in the output.
     *
     * This is to demonstrate that threads that have been initialized all execute the kernel function.
     */
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    printf("thread id = %d, a[%d] = %d\n", tid, tid, a[tid]);
}

__global__ void printArrayWithTidLimit(int *a, int N)
{
    /**
     * Here we use an if condition on the thread id.
     * execute kernel code if it is a valid memory location of *a
     */
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (tid < N)
    {
        printf("thread id = %d, a[%d] = %d\n", tid, tid, a[tid]);
    }
}

__global__ void printIndices()
{
    int tidx = (blockIdx.x * blockDim.x) + threadIdx.x;
    int tidy = (blockIdx.y * blockDim.y) + threadIdx.y;
    printf("blockIdx.x = %d, blockDim.x = %d, threadIdx.x = %d | tidx = %d\n", blockIdx.x, blockDim.x, threadIdx.x, tidx);
    printf("blockIdx.y = %d, blockDim.y = %d, threadIdx.y = %d | tidy = %d\n", blockIdx.y, blockDim.y, threadIdx.y, tidy);
}

int main(int argc, char **argv)
{
    int N = 10;
    vector<int> h_a(N);
    int *d_a;

    // threads per block
    int THREADS = 16;

    // blocks per grid
    int BLOCKS = (N + THREADS - 1) / THREADS;

    // generate random numbers
    for (int i = 0; i < N; i++)
    {
        h_a[i] = i;
    }

    cudaMalloc(&d_a, N * sizeof(int));
    cudaMemcpy(d_a, h_a.data(), N * sizeof(int), cudaMemcpyHostToDevice);

    printf("Printing array with tid limit\n");
    printArrayWithTidLimit<<<BLOCKS, THREADS>>>(d_a, N);
    cudaDeviceSynchronize();

    printf("Printing array without tid limit\n");
    printArrayWithoutTidLimit<<<BLOCKS, THREADS>>>(d_a, N);
    cudaDeviceSynchronize();

    printf("Printing indices\n");
    dim3 threads(2, 5);
    dim3 blocks(5, 5);
    printIndices<<<blocks, threads>>>();
    cudaDeviceSynchronize();

    return 0;
}