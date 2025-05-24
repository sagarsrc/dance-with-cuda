/**
 * Vector Addition using Unified Memory
 * reference: https://www.youtube.com/watch?v=84iwCupHW14&list=PLxNPSjHT5qvtYRVdNN1yDcdSl39uHV_sU&index=3
 *
 *
 *
 * Compile: nvcc vector_add_um_baseline.cu -o cu_vector_add_um_baseline.bin
 * Run: ./cu_vector_add_um_baseline.bin
 * Profiling: nvprof ./cu_vector_add_um_baseline.bin
 * Print GPU trace: nvprof  --print-gpu-trace ./cu_vector_add_um_baseline.bin
 */

#include <iostream>
#include <cuda_runtime.h>
#include <bits/stdc++.h>

using namespace std;

__global__ void vectorAdd(const int *__restrict a, const int *__restrict b,
                          int *__restrict c, int N)
{
    // Calculate global thread ID
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

    // Boundary check
    if (tid < N)
        c[tid] = a[tid] + b[tid];
}

void init_vector(int *a, int N)
{
    generate(a, a + N, []()
             { return rand() % 100; });
}

void check_result(int *a, int *b, int *c, int N)
{
    for (int i = 0; i < N; i++)
        assert(a[i] + b[i] == c[i]);
    cout << "Resultant vector is correct" << endl;
}

int main()
{

    // define array size
    const int N = 1 << 16;

    // declare unified memory pointers
    int *a, *b, *c;

    /**
     * NOTE: earlier we used cudaMalloc() to allocate memory on the device
     * now we use cudaMallocManaged() to allocate memory on the device and host
     * now we do not need to copy data from host to device and vice versa
     * we can directly use the pointers on both host and device
     * this is no longer required:
     * cudaMemcpy(c.data(), d_c, bytes, cudaMemcpyDeviceToHost);
     * these transfers will automatically happen when we use cudaMallocManaged()
     *
     */

    // allocate unified memory for these pointers
    cudaMallocManaged(&a, N * sizeof(int));
    cudaMallocManaged(&b, N * sizeof(int));
    cudaMallocManaged(&c, N * sizeof(int));

    // initialize array values
    init_vector(a, N);
    init_vector(b, N);

    // get number of blocks required to launch threads
    int threadsPerBlock = 1024;

    // calculate number of blocks required to launch threads
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // call kernel function
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(a, b, c, N);

    /**
     * cudaDeviceSynchronize() is a CUDA runtime API function
     * When called all the CPU threads will wait for all the preceeding CUDA operations to complete
     * e.g memory transfers, kernel launches etc
     * This also ensures that neither CPU nor the GPU is accessing the memory location simultaneously, hence avoiding race conditions
     * If you notice the output of nvprof, you will see that there are multiple page faults
     * This is because the data is not present on the device when the kernel is launched
     * so we need to page in the memory from host to device i.e bring in the data from host to device
     * Notice ==4539== Unified Memory profiling result section for checking number of page faults
     */

    // wait for all previous operations to complete
    cudaDeviceSynchronize();

    // check result
    check_result(a, b, c, N);

    // free memory
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);

    cout << "Completed successfully" << endl;
}

/**
nvprof   ./cu_vector_add_um_baseline.bin
==4539== NVPROF is profiling process 4539, command: ./cu_vector_add_um_baseline.bin
Resultant vector is correct
==4539== Profiling application: ./cu_vector_add_um_baseline.bin
==4539== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:  100.00%  458.30us         1  458.30us  458.30us  458.30us  vectorAdd(int const *, int const *, int*, int)
      API calls:   98.81%  306.69ms         3  102.23ms  3.4130us  306.66ms  cudaMallocManaged
                    0.93%  2.8957ms       114  25.400us      91ns  1.5833ms  cuDeviceGetAttribute
                    0.15%  461.91us         1  461.91us  461.91us  461.91us  cudaDeviceSynchronize
                    0.10%  299.28us         1  299.28us  299.28us  299.28us  cudaLaunchKernel
                    0.01%  35.697us         1  35.697us  35.697us  35.697us  cuDeviceGetName
                    0.00%  10.212us         1  10.212us  10.212us  10.212us  cuDeviceGetPCIBusId
                    0.00%  1.6750us         3     558ns     111ns  1.3240us  cuDeviceGetCount
                    0.00%     695ns         1     695ns     695ns     695ns  cuModuleGetLoadingMode
                    0.00%     513ns         1     513ns     513ns     513ns  cuDeviceTotalMem
                    0.00%     493ns         2     246ns     146ns     347ns  cuDeviceGet
                    0.00%     451ns         1     451ns     451ns     451ns  cuDeviceGetUuid

==4539== Unified Memory profiling result:
Device "Tesla T4 (0)"
   Count  Avg Size  Min Size  Max Size  Total Size  Total Time  Name
       8  64.000KB  4.0000KB  160.00KB  512.0000KB  65.72800us  Host To Device
      17  60.234KB  4.0000KB  380.00KB  1.000000MB  103.0400us  Device To Host
       2         -         -         -           -  445.1160us  Gpu page fault groups
Total CPU Page faults: 12
 */

/**
 nvprof  --print-gpu-trace ./cu_vector_add_um_baseline_baseline.bin
==4320== NVPROF is profiling process 4320, command: ./cu_vector_add_um_baseline.bin
Resultant vector is correct
==4320== Profiling application: ./cu_vector_add_um_baseline.bin
==4320== Profiling result:
   Start  Duration            Grid Size      Block Size     Regs*    SSMem*    DSMem*           Device   Context    Stream        Unified Memory  Virtual Address  Name
515.70ms         -                    -               -         -         -         -                -         -         -         PC 0x58344172   0x7b552e000000  [Unified Memory CPU page faults]
516.50ms         -                    -               -         -         -         -                -         -         -         PC 0x58344172   0x7b552e010000  [Unified Memory CPU page faults]
516.87ms         -                    -               -         -         -         -                -         -         -         PC 0x58344172   0x7b552e020000  [Unified Memory CPU page faults]
517.56ms         -                    -               -         -         -         -                -         -         -         PC 0x58344172   0x7b552e040000  [Unified Memory CPU page faults]
520.22ms  519.71us             (64 1 1)      (1024 1 1)        16        0B        0B     Tesla T4 (0)         1         7                     -                -  vectorAdd(int const *, int const *, int*, int) [128]
520.23ms  430.56us                    -               -         -         -         -     Tesla T4 (0)         -         -                     2   0x7b552e000000  [Unified Memory GPU page faults]
520.58ms  10.048us                    -               -         -         -         -     Tesla T4 (0)         -         -           64.000000KB   0x7b552e000000  [Unified Memory Memcpy HtoD]
520.59ms  3.9040us                    -               -         -         -         -     Tesla T4 (0)         -         -           16.000000KB   0x7b552e010000  [Unified Memory Memcpy HtoD]
520.60ms  8.0320us                    -               -         -         -         -     Tesla T4 (0)         -         -           64.000000KB   0x7b552e014000  [Unified Memory Memcpy HtoD]
520.60ms  2.8800us                    -               -         -         -         -     Tesla T4 (0)         -         -            4.000000KB   0x7b552e024000  [Unified Memory Memcpy HtoD]
520.61ms  3.4880us                    -               -         -         -         -     Tesla T4 (0)         -         -           12.000000KB   0x7b552e025000  [Unified Memory Memcpy HtoD]
520.61ms  10.944us                    -               -         -         -         -     Tesla T4 (0)         -         -           96.000000KB   0x7b552e028000  [Unified Memory Memcpy HtoD]
520.62ms  16.128us                    -               -         -         -         -     Tesla T4 (0)         -         -          160.000000KB   0x7b552e040000  [Unified Memory Memcpy HtoD]
520.64ms  10.784us                    -               -         -         -         -     Tesla T4 (0)         -         -           96.000000KB   0x7b552e068000  [Unified Memory Memcpy HtoD]
520.66ms  75.743us                    -               -         -         -         -     Tesla T4 (0)         -         -                     1   0x7b552e080000  [Unified Memory GPU page faults]
520.75ms         -                    -               -         -         -         -                -         -         -         PC 0x58343c77   0x7b552e000000  [Unified Memory CPU page faults]
520.80ms  1.8240us                    -               -         -         -         -     Tesla T4 (0)         -         -            4.000000KB   0x7b552e000000  [Unified Memory Memcpy DtoH]
520.80ms  5.9830us                    -               -         -         -         -     Tesla T4 (0)         -         -           60.000000KB   0x7b552e001000  [Unified Memory Memcpy DtoH]
520.88ms         -                    -               -         -         -         -                -         -         -         PC 0x58343c8d   0x7b552e040000  [Unified Memory CPU page faults]
520.89ms  1.8560us                    -               -         -         -         -     Tesla T4 (0)         -         -            4.000000KB   0x7b552e040000  [Unified Memory Memcpy DtoH]
520.89ms  5.9520us                    -               -         -         -         -     Tesla T4 (0)         -         -           60.000000KB   0x7b552e041000  [Unified Memory Memcpy DtoH]
520.97ms         -                    -               -         -         -         -                -         -         -         PC 0x58343ca5   0x7b552e080000  [Unified Memory CPU page faults]
520.99ms  1.8240us                    -               -         -         -         -     Tesla T4 (0)         -         -            4.000000KB   0x7b552e080000  [Unified Memory Memcpy DtoH]
520.99ms  5.9840us                    -               -         -         -         -     Tesla T4 (0)         -         -           60.000000KB   0x7b552e081000  [Unified Memory Memcpy DtoH]
521.10ms         -                    -               -         -         -         -                -         -         -         PC 0x58343c77   0x7b552e010000  [Unified Memory CPU page faults]
521.11ms  1.8570us                    -               -         -         -         -     Tesla T4 (0)         -         -            4.000000KB   0x7b552e010000  [Unified Memory Memcpy DtoH]
521.11ms  6.0150us                    -               -         -         -         -     Tesla T4 (0)         -         -           60.000000KB   0x7b552e011000  [Unified Memory Memcpy DtoH]
521.18ms         -                    -               -         -         -         -                -         -         -         PC 0x58343c8d   0x7b552e050000  [Unified Memory CPU page faults]
521.19ms  1.8880us                    -               -         -         -         -     Tesla T4 (0)         -         -            4.000000KB   0x7b552e050000  [Unified Memory Memcpy DtoH]
521.20ms  6.0800us                    -               -         -         -         -     Tesla T4 (0)         -         -           60.000000KB   0x7b552e051000  [Unified Memory Memcpy DtoH]
521.25ms         -                    -               -         -         -         -                -         -         -         PC 0x58343ca5   0x7b552e090000  [Unified Memory CPU page faults]
521.26ms  1.8240us                    -               -         -         -         -     Tesla T4 (0)         -         -            4.000000KB   0x7b552e090000  [Unified Memory Memcpy DtoH]
521.26ms  5.9520us                    -               -         -         -         -     Tesla T4 (0)         -         -           60.000000KB   0x7b552e091000  [Unified Memory Memcpy DtoH]
521.36ms         -                    -               -         -         -         -                -         -         -         PC 0x58343c77   0x7b552e020000  [Unified Memory CPU page faults]
521.38ms  1.8550us                    -               -         -         -         -     Tesla T4 (0)         -         -            4.000000KB   0x7b552e020000  [Unified Memory Memcpy DtoH]
521.38ms  10.752us                    -               -         -         -         -     Tesla T4 (0)         -         -          124.000000KB   0x7b552e021000  [Unified Memory Memcpy DtoH]
521.39ms  11.136us                    -               -         -         -         -     Tesla T4 (0)         -         -          128.000000KB   0x7b552e060000  [Unified Memory Memcpy DtoH]
521.48ms         -                    -               -         -         -         -                -         -         -         PC 0x58343ca5   0x7b552e0a0000  [Unified Memory CPU page faults]
521.51ms  1.8240us                    -               -         -         -         -     Tesla T4 (0)         -         -            4.000000KB   0x7b552e0a0000  [Unified Memory Memcpy DtoH]
521.51ms  30.144us                    -               -         -         -         -     Tesla T4 (0)         -         -          380.000000KB   0x7b552e0a1000  [Unified Memory Memcpy DtoH]

Regs: Number of registers used per CUDA thread. This number includes registers used internally by the CUDA driver and/or tools and can be more than what the compiler shows.
SSMem: Static shared memory allocated per CUDA block.
DSMem: Dynamic shared memory allocated per CUDA block.
 *
 */