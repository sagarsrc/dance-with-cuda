/**
 * Vector Addition
 * reference: https://www.youtube.com/watch?v=QVVTsLmMlwk&list=PLxNPSjHT5qvu4Q2UElj3HUCh2lpSooQWo&index=3
 * Each element in vector can be parallely added using CUDA.
 *
 * We need to specify which threads will handle the operations
 * For that there are some special variables:
 * - threadIdx: variable that contains the index of the thread in the block
 * - blockIdx: variable that contains the index of the block in the grid
 * - blockDim: variable that contains the number of threads in the block
 * - gridDim: variable that contains the number of blocks in the grid
 *
 *  [grid]
 *  [block, block, ...]
 *  [[thread, thread, ...], [thread, thread, ...], ...]
 *
 * Thread Block = CTA = Cooperative Thread Array (nvidia's term for block in CUDA)
 * Global TID = global thread id
 *
 * Global TID = blockIdx.x * blockDim.x + threadIdx.x
 *
 *
 * from previous outputs:
 * Device 0: "Tesla T4"
 *   Grid (highest level of hierarchy):
 *     Max grid dimensions: (2147483647, 65535, 65535)
 *     Max grid dimensions: (2^31 - 1, 2^16 - 1, 2^16 - 1)
 *
 *   Blocks (mid level):
 *     Max threads per block: 1024
 *     Max thread dimensions: (1024, 1024, 64)
 *     Max blocks per multiprocessor: 16
 *     Shared memory per block: 48.00 KB
 *
 *   Streaming Multiprocessors (hardware):
 *     Multiprocessor count: 40
 *     Warp size: 32 threads
 *     Total global memory: 14.56 GB
 *     Total constant memory: 64.00 KB
 *     Compute capability: 7.5
 *
 *
 * we have max threads per block = 1024
 *
 * Compilation: nvcc vector_add.cu -o vector_add.bin
 * Execution: ./vector_add.bin
 * Profiling: nvprof ./vector_add.bin
 * Print GPU trace: nvprof  --print-gpu-trace ./vector_add.bin
 */
#include <iostream>
#include <bits/stdc++.h>

using namespace std;

/**
 * - `*__restrict` is a qualifier that tells the compiler the pointers don't alias each other (don't point to the same memory location).
 * - This is a promise you make to the compiler that there's no overlap between the memory regions pointed to by these pointers.
 */
__global__ void vectorAdd(const int *__restrict a, const int *__restrict b,
                          int *__restrict c, int N)
{
    // Calculate global thread ID
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

    // Boundary check
    if (tid < N)
        c[tid] = a[tid] + b[tid];
}

void check_result(const vector<int> &a, const vector<int> &b, const vector<int> &c)
{
    for (int i = 0; i < a.size(); i++)
        if (a[i] + b[i] != c[i])
            throw runtime_error("Result is incorrect");
    cout << "Resultant vector is correct" << endl;
}

int main()
{
    // `constexpr` tells the compiler to evaluate the value at compile time
    // `1 << 16` is `2^16` which can be evaluated at compile time instead of runtime
    // this results in faster execution
    constexpr int N = 1 << 16;
    constexpr size_t bytes = sizeof(int) * N;

    // reserve memory for atleast N elements in advance
    // prevents reallocation of memory
    vector<int> a(N);
    a.reserve(N);
    vector<int> b(N);
    b.reserve(N);
    vector<int> c(N);
    c.reserve(N);

    // fill the vectors with random numbers between 0 and 100
    generate(a.begin(), a.end(), []()
             { return rand() % 100; });
    generate(b.begin(), b.end(), []()
             { return rand() % 100; });

    // allocate memory on device
    // `d_` for denoting device pointers
    // `h_` for denoting host pointers
    int *d_a, *d_b, *d_c;

    // cudaMalloc allocates memory on the device
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    // copy data from host to device
    cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice);

    // Get number of blocks required to launch threads
    // we need to have atleast as many threads as we have elements in the vector
    // so that each thread can handle one element
    // we can have multiple blocks to handle the work parallely
    int threadsPerBlock = 1024; // limited by the hardware
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);

    // copy data from device to host
    cudaMemcpy(c.data(), d_c, bytes, cudaMemcpyDeviceToHost);

    // check the result
    check_result(a, b, c);

    // free the allocated memory on the device
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}

/**
 * Run program with nvprof to see the performance
 * Inside GPU
 * - Here we can see multiple cuda api calls
 * - Host to Device memcpy takes maximum time
 * - Device to Host memcpy takes second maximum time
 * API calls:
 * - cudaMalloc takes maximum time
 * - cudaMemcpy takes second maximum time


# nvprof ./cu_vector_add.bin
==37665== NVPROF is profiling process 37665, command: ./cu_vector_add.bin
Resultant vector is correct
==37665== Profiling application: ./cu_vector_add.bin
==37665== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   63.38%  47.583us         2  23.791us  23.584us  23.999us  [CUDA memcpy HtoD]
                   29.37%  22.048us         1  22.048us  22.048us  22.048us  [CUDA memcpy DtoH]
                    7.25%  5.4400us         1  5.4400us  5.4400us  5.4400us  vectorAdd(int const *, int const *, int*, int)
      API calls:   98.44%  284.66ms         3  94.887ms  2.2980us  284.65ms  cudaMalloc
                    0.99%  2.8631ms       114  25.115us     100ns  1.5981ms  cuDeviceGetAttribute
                    0.40%  1.1541ms         1  1.1541ms  1.1541ms  1.1541ms  cudaLaunchKernel
                    0.10%  298.22us         3  99.407us  86.287us  108.44us  cudaMemcpy
                    0.05%  157.07us         3  52.356us  6.6780us  129.98us  cudaFree
                    0.00%  13.852us         1  13.852us  13.852us  13.852us  cuDeviceGetName
                    0.00%  8.6760us         1  8.6760us  8.6760us  8.6760us  cuDeviceGetPCIBusId
                    0.00%  1.4410us         3     480ns     118ns  1.1660us  cuDeviceGetCount
                    0.00%     601ns         2     300ns     107ns     494ns  cuDeviceGet
                    0.00%     357ns         1     357ns     357ns     357ns  cuDeviceTotalMem
                    0.00%     329ns         1     329ns     329ns     329ns  cuDeviceGetUuid
                    0.00%     242ns         1     242ns     242ns     242ns  cuModuleGetLoadingMode
 */

/**
 * To get global timeline of events we can use following command:
 # nvprof  --print-gpu-trace ./
cu_vector_add.bin
==38262== NVPROF is profiling process 38262, command: ./cu_vector_add.bin
Resultant vector is correct
==38262== Profiling application: ./cu_vector_add.bin
==38262== Profiling result:
   Start  Duration            Grid Size      Block Size     Regs*    SSMem*    DSMem*      Size  Throughput  SrcMemType  DstMemType           Device   Context    Stream  Name
465.87ms  26.016us                    -               -         -         -         -  256.00KB  9.3842GB/s    Pageable      Device     Tesla T4 (0)         1         7  [CUDA memcpy HtoD]
465.98ms  25.952us                    -               -         -         -         -  256.00KB  9.4074GB/s    Pageable      Device     Tesla T4 (0)         1         7  [CUDA memcpy HtoD]
466.21ms  4.9920us             (64 1 1)      (1024 1 1)        16        0B        0B         -           -           -           -     Tesla T4 (0)         1         7  vectorAdd(int const *, int const *, int*, int) [130]
466.23ms  22.016us                    -               -         -         -         -  256.00KB  11.089GB/s      Device    Pageable     Tesla T4 (0)         1         7  [CUDA memcpy DtoH]

Regs: Number of registers used per CUDA thread. This number includes registers used internally by the CUDA driver and/or tools and can be more than what the compiler shows.
SSMem: Static shared memory allocated per CUDA block.
DSMem: Dynamic shared memory allocated per CUDA block.
SrcMemType: The type of source memory accessed by memory operation/copy
DstMemType: The type of destination memory accessed by memory operation/copy

NOTE: Check actual vector addition time it is just ~5 microseconds, as compared to ~210 microseconds in C++

 */

/**
 * CUDA C/C++ → PTX (intermediate representation) → SASS (final machine code)
 *
 * To get ptx and saas code we can use following commands:
 * cuobjdump -ptx ./cu_vector_add.bin > ptx.asm
 * cuobjdump -saas ./cu_vector_add.bin > saas.asm
 *
 * What is PTX (Parallel Thread Execution)?
 * - PTX is a thread-level parallelism (TLP) programming model
 * - It is a low-level language that is used to write CUDA programs
 * - It is a subset of CUDA C
 * - It is a virtual machine that is used to run CUDA programs
 * - It is a intermediate representation of CUDA programs
 *
 * What is SASS (Source and Assembly)?
 * - It represents the final machine code instructions that run directly on the NVIDIA hardware
 * - SAAS varies between different NVIDIA GPU architectures (e.g., Ampere, Turing, Volta)
 * - It's the lowest-level representation of your CUDA program, after all optimizations
 *
 */