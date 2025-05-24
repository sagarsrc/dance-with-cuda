/**
 * NxN matrix with CUDA row and column printing
 * Printing rows and columns in matrix depends on number of threads in a block
 * if N = 2 and number of threads in a block is 2
 */

#include <cuda_runtime.h>
#include <bits/stdc++.h>

using namespace std;

// CUDA kernel to print only rows
__global__ void printRows(int *matrix, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N) {
        printf("Row %d: ", row);
        for (int i = 0; i < N; i++) {
            printf("%d ", matrix[row * N + i]);
        }
        printf("\n");
    }
}

// CUDA kernel to print only columns
__global__ void printColumns(int *matrix, int N) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < N) {
        printf("Col %d: ", col);
        for (int i = 0; i < N; i++) {
            printf("%d ", matrix[i * N + col]);
        }
        printf("\n");
    }
}

// Host function to print matrix
void printMatrix(int *a, int N, const string& name) {
    printf("%s:\n", name.c_str());
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {
            printf("%d ", a[i * N + j]);
        }
        printf("\n");
    }
    printf("\n");
}

// Function to assign values to matrix
void assignMatrix(vector<int> &matrix, int N, int startVal) {
    int val = startVal;
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {
            matrix[i * N + j] = val++;
        }
    }
}

int main() {
    int N = 3;  // NxN matrix
    vector<int> h_a(N * N);
    vector<int> h_b(N * N);

    // Assign values: Matrix A = [1,2; 3,4], Matrix B = [5,6; 7,8]
    assignMatrix(h_a, N, 1);
    assignMatrix(h_b, N, 5);

    // Print matrices on host
    printMatrix(h_a.data(), N, "Matrix A");
    printMatrix(h_b.data(), N, "Matrix B");

    // Allocate device memory
    int *d_a, *d_b;
    cudaMalloc(&d_a, N * N * sizeof(int));
    cudaMalloc(&d_b, N * N * sizeof(int));

    // Copy data to device
    cudaMemcpy(d_a, h_a.data(), N * N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), N * N * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernels with appropriate grid/block dimensions
    dim3 threads(N);  // N threads per block
    dim3 blocks(N);   // 1 block (since we only need N threads for NxN)

    printf("=== CUDA Row Printing ===\n");
    printf("Matrix A - Rows:\n");
    printRows<<<blocks, threads>>>(d_a, N);
    cudaDeviceSynchronize();

    printf("\nMatrix B - Rows:\n");
    printRows<<<blocks, threads>>>(d_b, N);
    cudaDeviceSynchronize();

    printf("\n=== CUDA Column Printing ===\n");
    printf("Matrix A - Columns:\n");
    printColumns<<<blocks, threads>>>(d_a, N);
    cudaDeviceSynchronize();

    printf("\nMatrix B - Columns:\n");
    printColumns<<<blocks, threads>>>(d_b, N);
    cudaDeviceSynchronize();

    // Cleanup
    cudaFree(d_a);
    cudaFree(d_b);

    return 0;
}


/*
Output:

Matrix A:
1 2
3 4

Matrix B:
5 6
7 8

=== CUDA Row Printing ===
Matrix A - Rows:
Row 0: Row 1: 1 3 2 4


Matrix B - Rows:
Row 0: Row 1: 5 7 6 8


=== CUDA Column Printing ===
Matrix A - Columns:
Col 0: Col 1: 1 2 3 4


Matrix B - Columns:
Col 0: Col 1: 5 6 7 8
*/