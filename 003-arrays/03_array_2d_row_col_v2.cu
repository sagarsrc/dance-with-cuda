/**
 * 2x2 matrix with CUDA row and column printing
 */

#include <cuda_runtime.h>
#include <bits/stdc++.h>

using namespace std;

// CUDA kernel to print only rows
__global__ void printRows(int *matrix, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N) {
        // Use atomic operations or serialize the printing
        for (int r = 0; r < N; r++) {
            if (row == r) {
                printf("Row %d: ", row);
                for (int i = 0; i < N; i++) {
                    printf("%d ", matrix[row * N + i]);
                }
                printf("\n");
            }
            __syncthreads();
        }
    }
}

// CUDA kernel to print only columns
__global__ void printColumns(int *matrix, int N) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < N) {
        // Serialize the printing to avoid interleaved output
        for (int c = 0; c < N; c++) {
            if (col == c) {
                printf("Col %d: ", col);
                for (int i = 0; i < N; i++) {
                    printf("%d ", matrix[i * N + col]);
                }
                printf("\n");
            }
            __syncthreads();
        }
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
    int N = 2;  // 2x2 matrix
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
    dim3 threads(2);  // 2 threads per block
    dim3 blocks(1);   // 1 block (since we only need 2 threads for 2x2)

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