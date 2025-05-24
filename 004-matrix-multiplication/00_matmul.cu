/**
 * Matrix Multiplication
 *
 * This program demonstrates square matrix multiplication using CUDA.
 *
 * Compilation: nvcc matmul.cu -o cu_matmul.bin -O3
 * Execution: ./cu_matmul.bin
 */

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <vector>
#include <bits/stdc++.h>

using namespace std;

__global__ void matrixMul(int *d_a, int *d_b, int *d_c, int N)
{
    /**
     * This is the most interesting part of the code.
     * When you first see this, you might think that how does CUDA know row and column from the blocks
     * Well the trick here is in understanding how the matrix is laid out in memory.
     *
     * Let's take a martrix A of size 2x2.
     *
     * A00  A01
     * A10  A11
     *
     * When you lay it out in memory, it looks like this:
     * A00,A01,A10,A11
     *
     * Similarly, matrix B of size 2x2 looks like this:
     * B00  B01
     * B10  B11
     *
     * When you lay it out in memory, it looks like this:
     * B00,B01,B10,B11
     *
     * When you multiply A and B, you get a result matrix C of size 2x2.
     *
     * C = A * B
     *
     * C00 C01
     * C10 C11
     *
     * C00 = A00*B00 + A01*B10
     * C01 = A00*B01 + A01*B11
     * C10 = A10*B00 + A11*B10
     * C11 = A10*B01 + A11*B11
     *
     * A00,A01,A10,A11 <<< How do you get row from this?
     * B00,B01,B10,B11 <<< How do you get column from this?
     *
     * first row of A is A00,A01 -> index 0,1
     * second row of A is A10,A11 -> index 2,3
     *
     * first column of B is B00,B10 -> index 0,2
     * second column of B is B01,B11 -> index 1,3
     *
     * we need to arrive at these indexes in the kernel during run time to calculate matmul
     *
     * For now let's stop here. And move to how CUDA executes this kernel.
     * When you define dim3 structs for block and grid dimensions,
     * you are essentially defining the number of blocks and threads per block.
     *
     * Once you call the kernel function using kernelfunc<<<blocks, threads>>>
     * every thread will have it's own unique threadIdx.x and threadIdx.y
     * and because threads lie in blocks it will also have it's own unique blockIdx.x and blockIdx.y
     *
     * Now every thread will start executing the kernel function.
     * This kernel function takes one row of matrix A, one column of matrix B
     * and calculates corresponding element of the result matrix.
     *
     * now you can see how row indices 0,1 and column indices 0,2 are calculated
     * using blockIdx.y, blockIdx.x, threadIdx.y, threadIdx.x
     *
     * Another thing to note here is:
     * - if you assign a large number of threads the threads will still execute this kernel function
     * - but will be blocked by if condition where we are specifying valid row and column indices
     *
     * Using inherent properties of threads, we are trying to determine:
     * 1. Which is a valid row number and column number that needs to be multiplied
     * 2. Once we have that, we can iterate sequentially over the elements of the row and column
     * 3. And calculate the result matrix element
     */
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N)
    {
        int sum = 0;
        for (int k = 0; k < N; k++)
        {
            sum += d_a[row * N + k] * d_b[k * N + col];
        }
        d_c[row * N + col] = sum;
    }
}

int main()
{

    int N = 20;
    vector<int> h_a(N * N);
    vector<int> h_b(N * N);
    vector<int> h_c(N * N);

    generate(h_a.begin(), h_a.end(), []()
             { return rand() % 100; });

    generate(h_b.begin(), h_b.end(), []()
             { return rand() % 100; });

    int *d_a, *d_b, *d_c;

    // Allocate memory on the device for matrices

    cudaMalloc(&d_a, N * N * sizeof(int));
    cudaMalloc(&d_b, N * N * sizeof(int));
    cudaMalloc(&d_c, N * N * sizeof(int));

    // Copy data from host to device
    cudaMemcpy(d_a, h_a.data(), N * N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), N * N * sizeof(int), cudaMemcpyHostToDevice);

    // Threads per block dimension
    int THREADS = 32;

    // Blocks per grid dimension (assumes THREADS divides N evenly)
    int BLOCKS = N / THREADS;

    // Use dim3 structs for block  and grid dimensions
    dim3 threads(THREADS, THREADS);
    dim3 blocks(BLOCKS, BLOCKS);

    // Launch kernel
    matrixMul<<<blocks, threads>>>(d_a, d_b, d_c, N);

    // Copy result back to host
    cudaMemcpy(h_c.data(), d_c, N * N * sizeof(int), cudaMemcpyDeviceToHost);

    return 0;
}