#include <stdio.h>
#include <iostream>
#include <cuda_runtime.h>
#include <string>
#include <vector>
#include <iomanip>

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

// Function to create a fixed-width string with content centered
std::string centerString(const std::string &text, int width)
{
    int padding = width - text.length();
    int padLeft = padding / 2;
    int padRight = padding - padLeft;
    return std::string(padLeft, ' ') + text + std::string(padRight, ' ');
}

// Function to create fixed-width string with left alignment and proper padding
std::string formatLine(const std::string &content, int width)
{
    if (content.length() >= width)
    {
        return content.substr(0, width);
    }
    return content + std::string(width - content.length(), ' ');
}

// Print a horizontal line with consistent characters
void printHLine(int width, char leftChar, char lineChar, char rightChar)
{
    std::cout << leftChar << std::string(width - 2, lineChar) << rightChar << std::endl;
}

// Print a content line with vertical borders
void printContentLine(const std::string &content, int width, char leftChar, char rightChar)
{
    std::cout << leftChar << formatLine(content, width - 2) << rightChar << std::endl;
}

// Print an empty line with just vertical borders
void printEmptyLine(int width, char borderChar)
{
    printContentLine("", width, borderChar, borderChar);
}

// Print a box section for blocks in a grid
void printGridBlocks(const std::string &label1, const std::string &label2, const std::string &label3)
{
    int blockWidth = 13;
    int spacing = 6;

    // Print labels
    std::cout << "|   " << formatLine(label1, blockWidth)
              << std::string(spacing, ' ') << formatLine(label2, blockWidth)
              << std::string(spacing, ' ') << formatLine(label3, blockWidth) << "     |" << std::endl;

    // Print top lines of blocks
    std::cout << "|   +" << std::string(blockWidth - 2, '-') << "+"
              << std::string(spacing, ' ') << "+" << std::string(blockWidth - 2, '-') << "+"
              << std::string(spacing, ' ') << "+" << std::string(blockWidth - 2, '-') << "+     |" << std::endl;

    // Print content lines
    for (int i = 0; i < 3; i++)
    {
        std::string middle = (i == 1 && label3.find("2") != std::string::npos) ? " . " : "   ";
        std::cout << "|   |" << std::string(blockWidth - 2, ' ') << "|"
                  << std::string(spacing, ' ') << "|" << std::string(blockWidth - 2, ' ') << "|"
                  << std::string(spacing, ' ') << "|" << formatLine(middle, blockWidth - 2) << "|     |" << std::endl;
    }

    // Print bottom lines of blocks
    std::cout << "|   +" << std::string(blockWidth - 2, '-') << "+"
              << std::string(spacing, ' ') << "+" << std::string(blockWidth - 2, '-') << "+"
              << std::string(spacing, ' ') << "+" << std::string(blockWidth - 2, '-') << "+     |" << std::endl;
}

// Print a title box
void printTitleBox(const std::string &title, const std::string &deviceName, int width)
{
    // Using equals sign for title box to differentiate it
    printHLine(width, '=', '=', '=');
    printContentLine("||" + centerString("CUDA EXECUTION HIERARCHY", width - 4) + "||", width, ' ', ' ');
    printContentLine("||" + centerString("Device: " + deviceName, width - 4) + "||", width, ' ', ' ');
    printHLine(width, '=', '=', '=');
    std::cout << std::endl;
}

// Print a section header
void printSectionHeader(const std::string &title, int width)
{
    std::cout << title << ":" << std::endl;
    printHLine(width, '+', '-', '+');
}

// Print a content box
void printContentBox(const std::string &title, int width, const std::vector<std::string> &content)
{
    printSectionHeader(title, width);

    for (const auto &line : content)
    {
        printContentLine(line, width, '|', '|');
    }

    printHLine(width, '+', '-', '+');
    std::cout << std::endl;
}

// Create content for the Grid Level section
std::vector<std::string> createGridContent(cudaDeviceProp prop)
{
    std::vector<std::string> content;

    content.push_back("");

    // Add block grid visualization
    // First row of blocks
    printGridBlocks("Block(0,0)", "Block(1,0)", "Block(2,0)");

    content.push_back("");

    // Second row of blocks
    printGridBlocks("Block(0,1)", "Block(1,1)", "Block(2,1)");

    content.push_back("");
    content.push_back("   ADDRESSING SPACE (theoretical limits):");

    char buffer[256];
    snprintf(buffer, sizeof(buffer), "   • Up to: %d x %d x %d blocks",
             prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    content.push_back(buffer);

    content.push_back("   • This is just the addressable space, not the");
    content.push_back("     number of blocks that run simultaneously");
    content.push_back("");

    return content;
}

// Print the CUDA execution hierarchy diagram
void printCudaHierarchy(cudaDeviceProp prop)
{
    const int BOX_WIDTH = 80;

    // Print the title box
    printTitleBox("CUDA EXECUTION HIERARCHY", prop.name, BOX_WIDTH);

    // GRID LEVEL SECTION
    {
        std::vector<std::string> gridContent;
        gridContent.push_back("");

        // First row of blocks
        gridContent.push_back("   Block(0,0)      Block(1,0)      Block(2,0)");
        gridContent.push_back("   +------------+  +------------+  +------------+");
        gridContent.push_back("   |            |  |            |  |            |");
        gridContent.push_back("   |            |  |            |  |     .      |");
        gridContent.push_back("   |            |  |            |  |            |");
        gridContent.push_back("   +------------+  +------------+  +------------+");
        gridContent.push_back("");

        // Second row of blocks
        gridContent.push_back("   Block(0,1)      Block(1,1)      Block(2,1)");
        gridContent.push_back("   +------------+  +------------+  +------------+");
        gridContent.push_back("   |            |  |            |  |            |");
        gridContent.push_back("   |            |  |            |  |     .      |");
        gridContent.push_back("   |            |  |            |  |            |");
        gridContent.push_back("   +------------+  +------------+  +------------+");
        gridContent.push_back("");

        gridContent.push_back("   ADDRESSING SPACE (theoretical limits):");

        char buffer[256];
        snprintf(buffer, sizeof(buffer), "   • Up to: %d x %d x %d blocks",
                 prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
        gridContent.push_back(buffer);

        gridContent.push_back("   • This is just the addressable space, not the");
        gridContent.push_back("     number of blocks that run simultaneously");
        gridContent.push_back("");

        printContentBox("GRID LEVEL (2D view of blocks)", BOX_WIDTH, gridContent);
    }

    // BLOCK LEVEL SECTION
    {
        std::vector<std::string> blockContent;
        blockContent.push_back("");
        blockContent.push_back("   Thread(0,0)   Thread(1,0)   Thread(2,0)   ...");
        blockContent.push_back("   Thread(0,1)   Thread(1,1)   Thread(2,1)   ...");
        blockContent.push_back("   Thread(0,2)   Thread(1,2)   Thread(2,2)   ...");
        blockContent.push_back("        ...          ...          ...        ...");
        blockContent.push_back("");

        char buffer[256];
        snprintf(buffer, sizeof(buffer), "   • Max threads per block: %d", prop.maxThreadsPerBlock);
        blockContent.push_back(buffer);

        snprintf(buffer, sizeof(buffer), "   • Thread dimensions: %d x %d x %d",
                 prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        blockContent.push_back(buffer);

        blockContent.push_back("   • Threads in a block share memory");

        snprintf(buffer, sizeof(buffer), "   • Shared memory per block: %.2f KB",
                 prop.sharedMemPerBlock / 1024.0f);
        blockContent.push_back(buffer);
        blockContent.push_back("");

        printContentBox("BLOCK LEVEL (single block, showing threads)", BOX_WIDTH, blockContent);
    }

    // HARDWARE LEVEL SECTION
    {
        std::vector<std::string> hwContent;
        hwContent.push_back("");
        hwContent.push_back("   SM 0        SM 1        SM 2        ...");
        hwContent.push_back("   +--------+  +--------+  +--------+");
        hwContent.push_back("   |Block   |  |Block   |  |Block   |");
        hwContent.push_back("   |Block   |  |Block   |  |Block   |");
        hwContent.push_back("   |Block   |  |Block   |  |Block   |");
        hwContent.push_back("   +--------+  +--------+  +--------+");
        hwContent.push_back("");
        hwContent.push_back("   PRACTICAL EXECUTION LIMITS:");

        char buffer[256];
        snprintf(buffer, sizeof(buffer), "   • Number of SMs: %d", prop.multiProcessorCount);
        hwContent.push_back(buffer);

        snprintf(buffer, sizeof(buffer), "   • Max blocks per SM: %d", prop.maxBlocksPerMultiProcessor);
        hwContent.push_back(buffer);

        snprintf(buffer, sizeof(buffer), "   • Warp size: %d threads", prop.warpSize);
        hwContent.push_back(buffer);

        snprintf(buffer, sizeof(buffer), "   • Total concurrent blocks: ~%d",
                 prop.multiProcessorCount * prop.maxBlocksPerMultiProcessor);
        hwContent.push_back(buffer);
        hwContent.push_back("");

        printContentBox("HARDWARE LEVEL (Streaming Multiprocessors)", BOX_WIDTH, hwContent);
    }

    // EXECUTION SUMMARY SECTION
    {
        std::vector<std::string> summaryContent;
        summaryContent.push_back("");
        summaryContent.push_back("  THE KEY POINT:");

        char buffer[256];
        snprintf(buffer, sizeof(buffer), "  • Theoretical grid space: %d x %d x %d blocks",
                 prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
        summaryContent.push_back(buffer);

        snprintf(buffer, sizeof(buffer), "  • But only ~%d blocks run simultaneously",
                 prop.multiProcessorCount * prop.maxBlocksPerMultiProcessor);
        summaryContent.push_back(buffer);

        summaryContent.push_back("");
        summaryContent.push_back("  1. Your kernel defines a grid of blocks");

        snprintf(buffer, sizeof(buffer), "  2. Each block contains up to %d threads", prop.maxThreadsPerBlock);
        summaryContent.push_back(buffer);

        snprintf(buffer, sizeof(buffer), "  3. The GPU has %d SMs (processors)", prop.multiProcessorCount);
        summaryContent.push_back(buffer);

        snprintf(buffer, sizeof(buffer), "  4. Each SM can process up to %d blocks at once",
                 prop.maxBlocksPerMultiProcessor);
        summaryContent.push_back(buffer);

        snprintf(buffer, sizeof(buffer), "  5. Max concurrent threads: ~%d",
                 prop.multiProcessorCount * prop.maxBlocksPerMultiProcessor * prop.maxThreadsPerBlock);
        summaryContent.push_back(buffer);

        summaryContent.push_back("");
        summaryContent.push_back("  THINK OF IT LIKE:");
        summaryContent.push_back("  • You can queue millions of blocks");

        snprintf(buffer, sizeof(buffer), "  • But only ~%d blocks run at once",
                 prop.multiProcessorCount * prop.maxBlocksPerMultiProcessor);
        summaryContent.push_back(buffer);

        summaryContent.push_back("  • The rest wait their turn");
        summaryContent.push_back("");

        printContentBox("EXECUTION SUMMARY", BOX_WIDTH, summaryContent);
    }
}

// Print device properties with improved formatting
void printDeviceProperties(cudaDeviceProp prop)
{
    printf("\n===== CUDA DEVICE PROPERTIES =====\n");

    // Grid level properties
    printf("\nADDRESSING LIMITS (Theoretical Maximum):\n");
    printf("\tGrid address space: (%d, %d, %d)\n",
           prop.maxGridSize[0],
           prop.maxGridSize[1],
           prop.maxGridSize[2]);
    printf("\t- X dimension: %d (2^31-1)\n", prop.maxGridSize[0]);
    printf("\t- Y dimension: %d (2^16-1)\n", prop.maxGridSize[1]);
    printf("\t- Z dimension: %d (2^16-1)\n", prop.maxGridSize[2]);
    printf("\tThread address space per block: (%d, %d, %d)\n",
           prop.maxThreadsDim[0],
           prop.maxThreadsDim[1],
           prop.maxThreadsDim[2]);
    printf("\t- Maximum threads per block: %d total\n", prop.maxThreadsPerBlock);

    // Hardware properties
    printf("\nEXECUTION RESOURCES (Practical Limits):\n");
    printf("\tHardware:\n");
    printf("\t- Streaming Multiprocessors (SMs): %d\n", prop.multiProcessorCount);
    printf("\t- Warp size: %d threads\n", prop.warpSize);
    printf("\t- Max blocks per SM: %d\n", prop.maxBlocksPerMultiProcessor);
    printf("\t- Shared memory per block: %.2f KB\n",
           static_cast<float>(prop.sharedMemPerBlock) / 1024);

    printf("\tConcurrent execution capacity:\n");
    printf("\t- Maximum concurrent blocks: ~%d (%d SMs × %d blocks)\n",
           prop.multiProcessorCount * prop.maxBlocksPerMultiProcessor,
           prop.multiProcessorCount, prop.maxBlocksPerMultiProcessor);
    printf("\t- Maximum concurrent threads: ~%d\n",
           prop.multiProcessorCount * prop.maxBlocksPerMultiProcessor * prop.maxThreadsPerBlock);

    printf("\tMemory:\n");
    printf("\t- Total global memory: %.2f GB\n",
           static_cast<float>(prop.totalGlobalMem) / (1024 * 1024 * 1024));
    printf("\t- Total constant memory: %.2f KB\n",
           static_cast<float>(prop.totalConstMem) / 1024);

    printf("\tCompute capability: %d.%d\n\n",
           prop.major, prop.minor);

    printf("NOTE: While you can address billions of blocks in theory,\n");
    printf("only ~%d blocks can execute concurrently on this device.\n",
           prop.multiProcessorCount * prop.maxBlocksPerMultiProcessor);
    printf("The rest wait in a queue.\n\n");
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

        // Print device properties
        printDeviceProperties(deviceProp);

        // Print ASCII diagram of CUDA hierarchy
        printCudaHierarchy(deviceProp);
    }

    return 0;
}