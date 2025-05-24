/**
 * NxN matrix with CUDA row and column printing
 * Printing rows and columns in matrix depends on number of threads in a block
 * if N = 2 and number of threads in a block is 2
 */

#include <cuda_runtime.h>
#include <bits/stdc++.h>

using namespace std;

// CUDA kernel to print only rows
__global__ void printRows(int *matrix, int num_rows, int num_cols) {

    int row_ix = blockIdx.x * blockDim.x + threadIdx.x;
    int col_ix = blockIdx.y * blockDim.y + threadIdx.y;

    if (row_ix < num_rows && col_ix < num_cols){
        printf("%d ", matrix[row_ix * num_cols + col_ix]);
        __syncthreads();
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
    int threads_per_block = N;
    int blocks_per_grid = ((N*N) + threads_per_block - 1 )/ threads_per_block;
    cout << "threads_per_block: " << threads_per_block << endl;
    cout << "blocks_per_grid: " << blocks_per_grid << endl;

    dim3 threads(threads_per_block, threads_per_block);
    dim3 blocks(blocks_per_grid);

    printf("=== CUDA Row Printing ===\n");
    printf("Matrix A - Rows:\n");
    printRows<<<blocks, threads>>>(d_a, N, N);
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
Why does this happen?

Matrix B - Rows:
Row 0: Row 1: 5 7 6 8


=== CUDA Column Printing ===
Matrix A - Columns:
Col 0: Col 1: 1 2 3 4


Matrix B - Columns:
Col 0: Col 1: 5 6 7 8
*/