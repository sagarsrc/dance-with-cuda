/**
 * Matrix Multiplication with Cache Tiling
 *
 * This program demonstrates square matrix multiplication using CUDA with cache
 * tiling.

 * reference:
 * [github](https://github.com/CoffeeBeforeArch/cuda_programming/blob/master/02_matrix_mul/tiled/mmul.cu)
 * [youtube](https://youtu.be/3xfyiWhtvZw?si=SEx_d2mCiuBs2iZL)
 * [lecture](https://engineering.purdue.edu/~smidkiff/ece563/NVidiaGPUTeachingToolkit/Mod4/Lecture-4-4-tiled-matrix-multiplication-kernel.pdf)
 *
 */

#include <algorithm>
#include <bits/stdc++.h>
#include <cassert>
#include <cstdlib>
#include <vector>

using namespace std;

const int N = 1 << 10;
const int SHMEM_SIZE = 1 << 10;

__global__ void matrixMulWithCacheTiling(int *a, int *b, int *c, int N) {
  // Compute each thread's global row and column index
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  // Statically allocated shared memory
  __shared__ int s_a[SHMEM_SIZE];
  __shared__ int s_b[SHMEM_SIZE];

  // Accumulate in temporary variable
  int tmp = 0;

  // Sweep tile across matrix
  for (int i = 0; i < N; i += blockDim.x) {
    // Load in elements for this tile
    s_a[threadIdx.y * blockDim.x + threadIdx.x] = a[row * N + i + threadIdx.x];
    s_b[threadIdx.y * blockDim.x + threadIdx.x] =
        b[i * N + threadIdx.y * N + col];

    // Wait for both tiles to be loaded in before doing computation
    __syncthreads();

    // Do matrix multiplication on the small matrix
    for (int j = 0; j < blockDim.x; j++) {
      tmp +=
          s_a[threadIdx.y * blockDim.x + j] * s_b[j * blockDim.x + threadIdx.x];
    }

    // Wait for all threads to finish using current tiles before loading in new
    // ones
    __syncthreads();
  }

  // Write back results
  c[row * N + col] = tmp;
}

// Check result on the CPU
void verify_result(vector<int> &a, vector<int> &b, vector<int> &c, int N) {
  // For every row...
  for (int i = 0; i < N; i++) {
    // For every column...
    for (int j = 0; j < N; j++) {
      // For every element in the row-column pair
      int tmp = 0;
      for (int k = 0; k < N; k++) {
        // Accumulate the partial results
        tmp += a[i * N + k] * b[k * N + j];
      }

      // Check against the CPU result
      assert(tmp == c[i * N + j]);
    }
  }
}

int main() {

  int N = 20;
  vector<int> h_a(N * N);
  vector<int> h_b(N * N);
  vector<int> h_c(N * N);

  generate(h_a.begin(), h_a.end(), []() { return rand() % 100; });

  generate(h_b.begin(), h_b.end(), []() { return rand() % 100; });

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
  matrixMulWithCacheTiling<<<blocks, threads>>>(d_a, d_b, d_c, N);

  // Copy result back to host
  cudaMemcpy(h_c.data(), d_c, N * N * sizeof(int), cudaMemcpyDeviceToHost);

  verify_result(h_a, h_b, h_c, N);

  free(d_a);
  free(d_b);
  free(d_c);

  return 0;
}