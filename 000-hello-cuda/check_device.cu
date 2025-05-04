/**
 * Check CUDA device properties
 * following is the output on a Tesla T4 GPU
 * This code was generated using Claude
 *
 * Compilation: nvcc check_device.cu -o cu_check_device.bin
 * Execution: ./cu_check_device.bin
 *
 * Found 1 CUDA device(s)
 *
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
 */
#include <stdio.h>
#include <cuda_runtime.h>

// Error handling function
#define checkCudaErrors(call)                                     \
    {                                                             \
        cudaError_t err = call;                                   \
        if (err != cudaSuccess)                                   \
        {                                                         \
            fprintf(stderr, "CUDA error in %s at line %d: %s\n",  \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                   \
        }                                                         \
    }

int main()
{
    int deviceCount = 0;
    checkCudaErrors(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0)
    {
        printf("No CUDA devices found\n");
        return 1;
    }

    printf("Found %d CUDA device(s)\n", deviceCount);

    // iterate over all devices
    for (int dev = 0; dev < deviceCount; dev++)
    {
        // instantiate device properties
        cudaDeviceProp deviceProp;

        checkCudaErrors(cudaGetDeviceProperties(&deviceProp, dev));

        printf("\nDevice %d: \"%s\"\n", dev, deviceProp.name);

        // Grid level properties
        printf("\nGrid Level (Highest Level):\n");
        printf("\tMax grid dimensions: (%d, %d, %d)\n",
               deviceProp.maxGridSize[0],
               deviceProp.maxGridSize[1],
               deviceProp.maxGridSize[2]);

        // Block level properties
        printf("\nBlock Level:\n");
        printf("\tMax threads per block: %d\n", deviceProp.maxThreadsPerBlock);
        printf("\tMax thread dimensions: (%d, %d, %d)\n",
               deviceProp.maxThreadsDim[0],
               deviceProp.maxThreadsDim[1],
               deviceProp.maxThreadsDim[2]);
        printf("\tMax blocks per multiprocessor: %d\n", deviceProp.maxBlocksPerMultiProcessor);
        printf("\tShared memory per block: %.2f KB\n",
               static_cast<float>(deviceProp.sharedMemPerBlock) / 1024);

        // Streaming Multiprocessor properties
        printf("\nStreaming Multiprocessor Level:\n");
        printf("\tMultiprocessor count: %d\n", deviceProp.multiProcessorCount);
        printf("\tWarp size: %d threads\n", deviceProp.warpSize);
        printf("\tTotal global memory: %.2f GB\n",
               static_cast<float>(deviceProp.totalGlobalMem) / (1024 * 1024 * 1024));
        printf("\tTotal constant memory: %.2f KB\n",
               static_cast<float>(deviceProp.totalConstMem) / 1024);
        printf("\tCompute capability: %d.%d\n",
               deviceProp.major, deviceProp.minor);
    }

    return 0;
}