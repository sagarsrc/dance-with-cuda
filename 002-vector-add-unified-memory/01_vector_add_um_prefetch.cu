/**
 * Vector Addition using Unified Memory
 * reference: https://www.youtube.com/watch?v=84iwCupHW14&list=PLxNPSjHT5qvtYRVdNN1yDcdSl39uHV_sU&index=3
 *
 *
 *
 * Compile: nvcc vector_add_um_prefetch.cu -o cu_vector_add_um_prefetch.bin
 * Run: ./cu_vector_add_um_prefetch.bin
 * Profiling: nvprof ./cu_vector_add_um_prefetch.bin
 * Print GPU trace: nvprof  --print-gpu-trace ./cu_vector_add_um_prefetch.bin
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
    size_t bytes = N * sizeof(int);

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
    cudaMallocManaged(&a, bytes);
    cudaMallocManaged(&b, bytes);
    cudaMallocManaged(&c, bytes);

    // initialize array values
    init_vector(a, N);
    init_vector(b, N);

    // get number of blocks required to launch threads
    int threadsPerBlock = 1024;

    // calculate number of blocks required to launch threads
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // prefetch data to the GPU

    // Get the device ID for prefetching calls
    int device_id;
    cudaGetDevice(&device_id);
    // cout << "Device ID: " << device_id << endl; // 0

    /**
     * The cudaMemAdvise() function is a CUDA runtime API call that provides hints to the CUDA driver
     * about how memory will be accessed, allowing it to optimize memory placement and migration.
     *
     * When you use the cudaMemAdviseSetReadMostly flag, you're telling the CUDA runtime system:
     * This memory will primarily be read from, not written to
     * This helps the CUDA runtime system to optimize memory placement and migration
     * It can lead to better performance by reducing memory contention and improving cache efficiency
     * Without cudaMemAdviseSetReadMostly, there are 12 page faults in baseline and 9 in prefetch
     * With cudaMemAdviseSetReadMostly, are only 4 page faults indicating smart placement of data on GPU
     */
    cudaMemAdvise(a, bytes, cudaMemAdviseSetReadMostly, device_id);
    cudaMemAdvise(b, bytes, cudaMemAdviseSetReadMostly, device_id);

    /**
     * Prefetch data to GPU before kernel execution
     * - devPtr: pointer to unified memory to prefetch
     * - count: number of bytes to prefetch
     * - dstDevice: GPU device ID to prefetch to
     * - stream: CUDA stream (0 for default stream)
     *
     * This helps avoid page faults during kernel execution by ensuring data is already on GPU
     * If you check the output of baseline nvprof and prefetch nvprof,
     * you will see that there is 1 page fault in prefetch whereas 2 page faults in baseline
     * In CPU there are 12 page faults in baseline and 9 in prefetch
     */

    // comment these lines to see the difference in profiling
    cudaMemPrefetchAsync(a, bytes, device_id);
    cudaMemPrefetchAsync(b, bytes, device_id);

    // call kernel function
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(a, b, c, N);

    /**
     * cudaDeviceSynchronize() is a CUDA runtime API function
     * When called all the CPU threads will wait for all the preceeding CUDA operations to complete
     * Earlier we used cudaMemcpy() function to explicitly specify that certain data is required to be on the device / host
     * Here we are using cudaMemPrefetchAsync() to prefetch data to the GPU
     * This way we ensure that the data needed by the GPU is already on the GPU
     */

    // wait for all previous operations to complete
    cudaDeviceSynchronize();

    /**
     * prefetch data back to CPU
     * cudaCpuDeviceId is the device ID for the CPU which is predefined
     * while vectorAdd() is running on the GPU, the CPU is waiting for it to complete
     * so we prefetch the results which are ready on the GPU to the CPU without waiting for the entire operation to complete
     */
    cudaMemPrefetchAsync(c, bytes, cudaCpuDeviceId);

    // check result
    check_result(a, b, c, N);

    // free memory
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);

    cout << "Completed successfully" << endl;
}
/*##########################################################################*/
/*################ if you run without cudaMemAdvise ########################*/
/*##########################################################################*/

/**
nvprof ./vector_add_um_prefetch.bin
==9579== NVPROF is profiling process 9579, command: ./vector_add_um_prefetch.bin
Resultant vector is correct
Completed successfully
==9579== Profiling application: ./vector_add_um_prefetch.bin
==9579== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:  100.00%  236.41us         1  236.41us  236.41us  236.41us  vectorAdd(int const *, int const *, int*, int)
      API calls:   98.62%  300.51ms         3  100.17ms  3.5900us  300.49ms  cudaMallocManaged
                    0.98%  2.9786ms       114  26.127us      90ns  1.6524ms  cuDeviceGetAttribute
                    0.15%  471.32us         3  157.11us  118.37us  229.93us  cudaMemPrefetchAsync
                    0.09%  284.09us         1  284.09us  284.09us  284.09us  cudaLaunchKernel
                    0.08%  238.94us         1  238.94us  238.94us  238.94us  cudaDeviceSynchronize
                    0.06%  198.02us         3  66.006us  16.395us  147.61us  cudaFree
                    0.00%  14.238us         1  14.238us  14.238us  14.238us  cudaGetDevice
                    0.00%  12.106us         1  12.106us  12.106us  12.106us  cuDeviceGetName
                    0.00%  10.229us         1  10.229us  10.229us  10.229us  cuDeviceGetPCIBusId
                    0.00%     805ns         3     268ns     103ns     569ns  cuDeviceGetCount
                    0.00%     546ns         2     273ns      99ns     447ns  cuDeviceGet
                    0.00%     358ns         1     358ns     358ns     358ns  cuDeviceTotalMem
                    0.00%     237ns         1     237ns     237ns     237ns  cuDeviceGetUuid
                    0.00%     203ns         1     203ns     203ns     203ns  cuModuleGetLoadingMode

==9579== Unified Memory profiling result:
Device "Tesla T4 (0)"
   Count  Avg Size  Min Size  Max Size  Total Size  Total Time  Name
       2  256.00KB  256.00KB  256.00KB  512.0000KB  50.11200us  Host To Device
      13  78.769KB  4.0000KB  256.00KB  1.000000MB  97.69600us  Device To Host
       1         -         -         -           -  228.1900us  Gpu page fault groups
Total CPU Page faults: 9
 */

/**
 nvprof --print-gpu-trace ./vector_add_um_prefetch.bin
==9733== NVPROF is profiling process 9733, command: ./vector_add_um_prefetch.bin
Resultant vector is correct
Completed successfully
==9733== Profiling application: ./vector_add_um_prefetch.bin
==9733== Profiling result:
   Start  Duration            Grid Size      Block Size     Regs*    SSMem*    DSMem*           Device   Context    Stream        Unified Memory  Virtual Address  Name
488.05ms         -                    -               -         -         -         -                -         -         -         PC 0x46de3225   0x7612d2000000  [Unified Memory CPU page faults]
488.82ms         -                    -               -         -         -         -                -         -         -         PC 0x46de3225   0x7612d2010000  [Unified Memory CPU page faults]
489.24ms         -                    -               -         -         -         -                -         -         -         PC 0x46de3225   0x7612d2020000  [Unified Memory CPU page faults]
489.96ms         -                    -               -         -         -         -                -         -         -         PC 0x46de3225   0x7612d2040000  [Unified Memory CPU page faults]
491.48ms  25.952us                    -               -         -         -         -     Tesla T4 (0)         -         -          256.000000KB   0x7612d2000000  [Unified Memory Memcpy HtoD]
491.90ms  23.967us                    -               -         -         -         -     Tesla T4 (0)         -         -          256.000000KB   0x7612d2040000  [Unified Memory Memcpy HtoD]
492.14ms  222.81us             (64 1 1)      (1024 1 1)        16        0B        0B     Tesla T4 (0)         1         7                     -                -  vectorAdd(int const *, int const *, int*, int) [131]
492.14ms  214.81us                    -               -         -         -         -     Tesla T4 (0)         -         -                     1   0x7612d2080000  [Unified Memory GPU page faults]
492.43ms  20.736us                    -               -         -         -         -     Tesla T4 (0)         -         -          256.000000KB   0x7612d2080000  [Unified Memory Memcpy DtoH]
492.48ms         -                    -               -         -         -         -                -         -         -         PC 0x46de2c77   0x7612d2000000  [Unified Memory CPU page faults]
492.50ms  1.8250us                    -               -         -         -         -     Tesla T4 (0)         -         -            4.000000KB   0x7612d2000000  [Unified Memory Memcpy DtoH]
492.50ms  6.0150us                    -               -         -         -         -     Tesla T4 (0)         -         -           60.000000KB   0x7612d2001000  [Unified Memory Memcpy DtoH]
492.57ms         -                    -               -         -         -         -                -         -         -         PC 0x46de2c8d   0x7612d2040000  [Unified Memory CPU page faults]
492.58ms  1.8880us                    -               -         -         -         -     Tesla T4 (0)         -         -            4.000000KB   0x7612d2040000  [Unified Memory Memcpy DtoH]
492.58ms  5.9520us                    -               -         -         -         -     Tesla T4 (0)         -         -           60.000000KB   0x7612d2041000  [Unified Memory Memcpy DtoH]
492.71ms         -                    -               -         -         -         -                -         -         -         PC 0x46de2c77   0x7612d2010000  [Unified Memory CPU page faults]
492.72ms  1.8880us                    -               -         -         -         -     Tesla T4 (0)         -         -            4.000000KB   0x7612d2010000  [Unified Memory Memcpy DtoH]
492.73ms  5.9840us                    -               -         -         -         -     Tesla T4 (0)         -         -           60.000000KB   0x7612d2011000  [Unified Memory Memcpy DtoH]
492.78ms         -                    -               -         -         -         -                -         -         -         PC 0x46de2c8d   0x7612d2050000  [Unified Memory CPU page faults]
492.79ms  1.8550us                    -               -         -         -         -     Tesla T4 (0)         -         -            4.000000KB   0x7612d2050000  [Unified Memory Memcpy DtoH]
492.80ms  6.0480us                    -               -         -         -         -     Tesla T4 (0)         -         -           60.000000KB   0x7612d2051000  [Unified Memory Memcpy DtoH]
492.91ms         -                    -               -         -         -         -                -         -         -         PC 0x46de2c77   0x7612d2020000  [Unified Memory CPU page faults]
492.95ms  1.8560us                    -               -         -         -         -     Tesla T4 (0)         -         -            4.000000KB   0x7612d2020000  [Unified Memory Memcpy DtoH]
492.95ms  10.784us                    -               -         -         -         -     Tesla T4 (0)         -         -          124.000000KB   0x7612d2021000  [Unified Memory Memcpy DtoH]
492.96ms  11.136us                    -               -         -         -         -     Tesla T4 (0)         -         -          128.000000KB   0x7612d2060000  [Unified Memory Memcpy DtoH]
492.97ms  21.216us                    -               -         -         -         -     Tesla T4 (0)         -         -          256.000000KB   0x7612d20c0000  [Unified Memory Memcpy DtoH]

Regs: Number of registers used per CUDA thread. This number includes registers used internally by the CUDA driver and/or tools and can be more than what the compiler shows.
SSMem: Static shared memory allocated per CUDA block.
DSMem: Dynamic shared memory allocated per CUDA block.
 *
 */

/*##########################################################################*/
/*################ if you run with cudaMemAdvise ###########################*/
/*##########################################################################*/

/**
 *
 * nvprof ./vector_add_um_prefetch.bin
==10599== NVPROF is profiling process 10599, command: ./vector_add_um_prefetch.bin
Resultant vector is correct
Completed successfully
==10599== Profiling application: ./vector_add_um_prefetch.bin
==10599== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:  100.00%  338.62us         1  338.62us  338.62us  338.62us  vectorAdd(int const *, int const *, int*, int)
      API calls:   98.59%  307.91ms         3  102.64ms  3.3010us  307.88ms  cudaMallocManaged
                    0.95%  2.9794ms       114  26.135us      91ns  1.6356ms  cuDeviceGetAttribute
                    0.11%  340.96us         1  340.96us  340.96us  340.96us  cudaDeviceSynchronize
                    0.11%  328.49us         3  109.50us  46.450us  158.15us  cudaMemPrefetchAsync
                    0.10%  303.86us         1  303.86us  303.86us  303.86us  cudaLaunchKernel
                    0.08%  257.51us         3  85.836us  25.847us  181.71us  cudaFree
                    0.05%  144.64us         2  72.322us  46.864us  97.780us  cudaMemAdvise
                    0.01%  18.542us         1  18.542us  18.542us  18.542us  cuDeviceGetName
                    0.00%  9.9220us         1  9.9220us  9.9220us  9.9220us  cuDeviceGetPCIBusId
                    0.00%  3.9460us         1  3.9460us  3.9460us  3.9460us  cudaGetDevice
                    0.00%  1.8590us         3     619ns      99ns  1.3540us  cuDeviceGetCount
                    0.00%     441ns         2     220ns     107ns     334ns  cuDeviceGet
                    0.00%     404ns         1     404ns     404ns     404ns  cuModuleGetLoadingMode
                    0.00%     359ns         1     359ns     359ns     359ns  cuDeviceTotalMem
                    0.00%     184ns         1     184ns     184ns     184ns  cuDeviceGetUuid

==10599== Unified Memory profiling result:
Device "Tesla T4 (0)"
   Count  Avg Size  Min Size  Max Size  Total Size  Total Time  Name
       2  256.00KB  256.00KB  256.00KB  512.0000KB  50.01600us  Host To Device
       1  256.00KB  256.00KB  256.00KB  256.0000KB  20.76700us  Device To Host
       1         -         -         -           -  330.7170us  Gpu page fault groups
Total CPU Page faults: 4
 */

/**
 * nvprof --print-gpu-trace ./vector_add_um_prefetch.bin
==10916== NVPROF is profiling process 10916, command: ./vector_add_um_prefetch.bin
Resultant vector is correct
Completed successfully
==10916== Profiling application: ./vector_add_um_prefetch.bin
==10916== Profiling result:
   Start  Duration            Grid Size      Block Size     Regs*    SSMem*    DSMem*           Device   Context    Stream        Unified Memory  Virtual Address  Name
479.00ms         -                    -               -         -         -         -                -         -         -         PC 0xcc43d259   0x79c46e000000  [Unified Memory CPU page faults]
479.82ms         -                    -               -         -         -         -                -         -         -         PC 0xcc43d259   0x79c46e010000  [Unified Memory CPU page faults]
480.18ms         -                    -               -         -         -         -                -         -         -         PC 0xcc43d259   0x79c46e020000  [Unified Memory CPU page faults]
480.87ms         -                    -               -         -         -         -                -         -         -         PC 0xcc43d259   0x79c46e040000  [Unified Memory CPU page faults]
482.45ms  25.856us                    -               -         -         -         -     Tesla T4 (0)         -         -          256.000000KB   0x79c46e000000  [Unified Memory Memcpy HtoD]
482.53ms  23.871us                    -               -         -         -         -     Tesla T4 (0)         -         -          256.000000KB   0x79c46e040000  [Unified Memory Memcpy HtoD]
482.84ms  290.27us             (64 1 1)      (1024 1 1)        16        0B        0B     Tesla T4 (0)         1         7                     -                -  vectorAdd(int const *, int const *, int*, int) [133]
482.85ms  283.07us                    -               -         -         -         -     Tesla T4 (0)         -         -                     1   0x79c46e080000  [Unified Memory GPU page faults]
483.22ms  20.768us                    -               -         -         -         -     Tesla T4 (0)         -         -          256.000000KB   0x79c46e080000  [Unified Memory Memcpy DtoH]

Regs: Number of registers used per CUDA thread. This number includes registers used internally by the CUDA driver and/or tools and can be more than what the compiler shows.
SSMem: Static shared memory allocated per CUDA block.
DSMem: Dynamic shared memory allocated per CUDA block.
 */