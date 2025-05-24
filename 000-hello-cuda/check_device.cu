#include <stdio.h>
#include <iostream>
#include <cuda_runtime.h>
#include <string>
#include <vector>
#include <algorithm>
#include <sstream>

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

// Safe string truncation and padding
std::string safeFormat(const std::string &text, int width, bool center = false)
{
    if (width <= 0) return "";

    std::string result = text;

    // Truncate if too long
    if (result.length() > static_cast<size_t>(width))
    {
        result = result.substr(0, width - 3) + "...";
    }

    // Pad to exact width
    if (center)
    {
        int padding = width - result.length();
        int padLeft = padding / 2;
        int padRight = padding - padLeft;
        result = std::string(padLeft, ' ') + result + std::string(padRight, ' ');
    }
    else
    {
        result += std::string(width - result.length(), ' ');
    }

    return result;
}

// Print horizontal line with exact width
void printHorizontalLine(int width, char corner, char line)
{
    std::cout << corner << std::string(width - 2, line) << corner << std::endl;
}

// Print content line with exact width and borders
void printContentLine(const std::string &content, int width)
{
    std::string formatted = safeFormat(content, width - 2);
    std::cout << "|" << formatted << "|" << std::endl;
}

// Print empty line
void printEmptyLine(int width)
{
    printContentLine("", width);
}

// Print centered title line
void printTitleLine(const std::string &title, int width)
{
    std::string formatted = safeFormat(title, width - 4, true);
    std::cout << "||" << formatted << "||" << std::endl;
}

// Print title box with double borders
void printTitleBox(const std::string &title, const std::string &deviceName, int width)
{
    printHorizontalLine(width, '=', '=');
    printTitleLine("CUDA EXECUTION HIERARCHY", width);
    printTitleLine("Device: " + deviceName, width);
    printHorizontalLine(width, '=', '=');
    std::cout << std::endl;
}

// Print section header
void printSectionHeader(const std::string &title, int width)
{
    printHorizontalLine(width, '+', '-');
    printContentLine(" " + title, width);
    printHorizontalLine(width, '+', '-');
}

// Print section footer
void printSectionFooter(int width)
{
    printHorizontalLine(width, '+', '-');
    std::cout << std::endl;
}

// Create a visual block representation
void printBlock(const std::string &label, int blockWidth, bool showDot = false)
{
    // Block label
    std::cout << safeFormat(label, blockWidth);

    // Block top border
    std::cout << std::endl << "+" << std::string(blockWidth - 2, '-') << "+";

    // Block content (3 lines)
    for (int i = 0; i < 3; i++)
    {
        std::cout << std::endl << "|";
        if (i == 1 && showDot)
        {
            std::string dotLine = safeFormat(" . ", blockWidth - 2, true);
            std::cout << dotLine;
        }
        else
        {
            std::cout << std::string(blockWidth - 2, ' ');
        }
        std::cout << "|";
    }

    // Block bottom border
    std::cout << std::endl << "+" << std::string(blockWidth - 2, '-') << "+";
}

// Print grid blocks in a row
void printGridRow(const std::vector<std::string> &labels, int width)
{
    const int blockWidth = 15;
    const int spacing = 4;
    const int numBlocks = 3;

    // Calculate if blocks fit
    int totalBlockWidth = numBlocks * blockWidth + (numBlocks - 1) * spacing;
    int leftMargin = (width - 2 - totalBlockWidth) / 2;

    std::string margin = std::string(std::max(1, leftMargin), ' ');

    // Print each line of the blocks
    for (int line = 0; line < 6; line++) // 6 lines per block (label + top + 3 content + bottom)
    {
        std::cout << "|" << margin;

        for (int block = 0; block < numBlocks && block < labels.size(); block++)
        {
            if (block > 0) std::cout << std::string(spacing, ' ');

            switch (line)
            {
                case 0: // Label
                    std::cout << safeFormat(labels[block], blockWidth);
                    break;
                case 1: // Top border
                    std::cout << "+" << std::string(blockWidth - 2, '-') << "+";
                    break;
                case 2: // Content line 1
                case 4: // Content line 3
                    std::cout << "|" << std::string(blockWidth - 2, ' ') << "|";
                    break;
                case 3: // Content line 2 (middle, may have dot)
                    std::cout << "|";
                    if (labels[block].find("2") != std::string::npos)
                    {
                        std::cout << safeFormat(" . ", blockWidth - 2, true);
                    }
                    else
                    {
                        std::cout << std::string(blockWidth - 2, ' ');
                    }
                    std::cout << "|";
                    break;
                case 5: // Bottom border
                    std::cout << "+" << std::string(blockWidth - 2, '-') << "+";
                    break;
            }
        }

        // Fill remaining space
        int usedWidth = margin.length() + totalBlockWidth;
        int remainingWidth = width - 2 - usedWidth;
        std::cout << std::string(std::max(0, remainingWidth), ' ') << "|" << std::endl;
    }
}

// Create formatted number with commas
std::string formatNumber(long long number)
{
    std::string str = std::to_string(number);
    std::string result;
    int count = 0;

    for (int i = str.length() - 1; i >= 0; i--)
    {
        if (count == 3)
        {
            result = "," + result;
            count = 0;
        }
        result = str[i] + result;
        count++;
    }

    return result;
}

// Print device properties in a structured way
void printDeviceProperties(const cudaDeviceProp &prop, int width)
{
    std::vector<std::string> content;

    content.push_back("");
    content.push_back(" ADDRESSING LIMITS (Theoretical Maximum):");
    content.push_back("");

    char buffer[256];
    snprintf(buffer, sizeof(buffer), " Grid dimensions: %s x %s x %s blocks",
             formatNumber(prop.maxGridSize[0]).c_str(),
             formatNumber(prop.maxGridSize[1]).c_str(),
             formatNumber(prop.maxGridSize[2]).c_str());
    content.push_back(buffer);

    snprintf(buffer, sizeof(buffer), " Thread dimensions: %d x %d x %d per block",
             prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    content.push_back(buffer);

    snprintf(buffer, sizeof(buffer), " Max threads per block: %s",
             formatNumber(prop.maxThreadsPerBlock).c_str());
    content.push_back(buffer);

    content.push_back("");
    content.push_back(" HARDWARE RESOURCES (Practical Limits):");
    content.push_back("");

    snprintf(buffer, sizeof(buffer), " Streaming Multiprocessors: %d",
             prop.multiProcessorCount);
    content.push_back(buffer);

    snprintf(buffer, sizeof(buffer), " Max blocks per SM: %d",
             prop.maxBlocksPerMultiProcessor);
    content.push_back(buffer);

    snprintf(buffer, sizeof(buffer), " Warp size: %d threads", prop.warpSize);
    content.push_back(buffer);

    long long maxConcurrentBlocks = (long long)prop.multiProcessorCount * prop.maxBlocksPerMultiProcessor;
    snprintf(buffer, sizeof(buffer), " Max concurrent blocks: %s",
             formatNumber(maxConcurrentBlocks).c_str());
    content.push_back(buffer);

    long long maxConcurrentThreads = maxConcurrentBlocks * prop.maxThreadsPerBlock;
    snprintf(buffer, sizeof(buffer), " Max concurrent threads: %s",
             formatNumber(maxConcurrentThreads).c_str());
    content.push_back(buffer);

    content.push_back("");
    content.push_back(" MEMORY:");
    content.push_back("");

    snprintf(buffer, sizeof(buffer), " Global memory: %.1f GB",
             static_cast<double>(prop.totalGlobalMem) / (1024.0 * 1024.0 * 1024.0));
    content.push_back(buffer);

    snprintf(buffer, sizeof(buffer), " Shared memory per block: %.1f KB",
             static_cast<double>(prop.sharedMemPerBlock) / 1024.0);
    content.push_back(buffer);

    snprintf(buffer, sizeof(buffer), " Constant memory: %.1f KB",
             static_cast<double>(prop.totalConstMem) / 1024.0);
    content.push_back(buffer);

    content.push_back("");
    snprintf(buffer, sizeof(buffer), " Compute capability: %d.%d",
             prop.major, prop.minor);
    content.push_back(buffer);
    content.push_back("");

    // Print the content
    printSectionHeader("DEVICE PROPERTIES", width);
    for (const auto &line : content)
    {
        printContentLine(line, width);
    }
    printSectionFooter(width);
}

// Print the CUDA execution hierarchy
void printCudaHierarchy(const cudaDeviceProp &prop, int width)
{
    // GRID LEVEL
    printSectionHeader("GRID LEVEL - Layout of Blocks", width);
    printEmptyLine(width);

    // First row of blocks
    std::vector<std::string> row1 = {"Block(0,0)", "Block(1,0)", "Block(2,0)"};
    printGridRow(row1, width);

    printEmptyLine(width);

    // Second row of blocks
    std::vector<std::string> row2 = {"Block(0,1)", "Block(1,1)", "Block(2,1)"};
    printGridRow(row2, width);

    printEmptyLine(width);
    printContentLine(" Note: Blocks are scheduled and executed by the hardware", width);
    printContentLine(" as resources become available.", width);
    printEmptyLine(width);
    printSectionFooter(width);

    // BLOCK LEVEL
    printSectionHeader("BLOCK LEVEL - Threads within a Block", width);
    printEmptyLine(width);
    printContentLine("   Thread(0,0)  Thread(1,0)  Thread(2,0)  Thread(3,0) ...", width);
    printContentLine("   Thread(0,1)  Thread(1,1)  Thread(2,1)  Thread(3,1) ...", width);
    printContentLine("   Thread(0,2)  Thread(1,2)  Thread(2,2)  Thread(3,2) ...", width);
    printContentLine("   Thread(0,3)  Thread(1,3)  Thread(2,3)  Thread(3,3) ...", width);
    printContentLine("        ...          ...          ...          ...     ...", width);
    printEmptyLine(width);

    char buffer[256];
    snprintf(buffer, sizeof(buffer), " Threads per block: up to %s",
             formatNumber(prop.maxThreadsPerBlock).c_str());
    printContentLine(buffer, width);

    snprintf(buffer, sizeof(buffer), " Threads share %.1f KB of shared memory",
             static_cast<double>(prop.sharedMemPerBlock) / 1024.0);
    printContentLine(buffer, width);

    snprintf(buffer, sizeof(buffer), " Threads execute in warps of %d", prop.warpSize);
    printContentLine(buffer, width);
    printEmptyLine(width);
    printSectionFooter(width);

    // HARDWARE LEVEL
    printSectionHeader("HARDWARE LEVEL - Streaming Multiprocessors", width);
    printEmptyLine(width);
    printContentLine("     SM 0           SM 1           SM 2         ...", width);
    printContentLine("   +----------+   +----------+   +----------+", width);
    printContentLine("   | Block A  |   | Block D  |   | Block G  |", width);
    printContentLine("   | Block B  |   | Block E  |   | Block H  |", width);
    printContentLine("   | Block C  |   | Block F  |   | Block I  |", width);
    printContentLine("   +----------+   +----------+   +----------+", width);
    printEmptyLine(width);

    snprintf(buffer, sizeof(buffer), " Total SMs: %d", prop.multiProcessorCount);
    printContentLine(buffer, width);

    snprintf(buffer, sizeof(buffer), " Max blocks per SM: %d", prop.maxBlocksPerMultiProcessor);
    printContentLine(buffer, width);

    long long totalConcurrentBlocks = (long long)prop.multiProcessorCount * prop.maxBlocksPerMultiProcessor;
    snprintf(buffer, sizeof(buffer), " Total concurrent blocks: %s",
             formatNumber(totalConcurrentBlocks).c_str());
    printContentLine(buffer, width);
    printEmptyLine(width);
    printSectionFooter(width);

    // EXECUTION SUMMARY
    printSectionHeader("EXECUTION SUMMARY", width);
    printEmptyLine(width);
    printContentLine(" KEY CONCEPT:", width);
    printEmptyLine(width);

    long long theoreticalBlocks = (long long)prop.maxGridSize[0] * prop.maxGridSize[1] * prop.maxGridSize[2];
    snprintf(buffer, sizeof(buffer), " Theoretical grid capacity: %s blocks",
             formatNumber(theoreticalBlocks).c_str());
    printContentLine(buffer, width);

    snprintf(buffer, sizeof(buffer), " Actual concurrent execution: %s blocks",
             formatNumber(totalConcurrentBlocks).c_str());
    printContentLine(buffer, width);

    printEmptyLine(width);
    printContentLine(" EXECUTION FLOW:", width);
    printContentLine(" 1. Your kernel launches a grid of blocks", width);
    printContentLine(" 2. GPU scheduler assigns blocks to available SMs", width);
    printContentLine(" 3. Each SM processes its assigned blocks", width);
    printContentLine(" 4. When a block finishes, SM gets a new block", width);
    printContentLine(" 5. Process continues until all blocks complete", width);
    printEmptyLine(width);
    printContentLine(" ANALOGY: Think of it like a restaurant:", width);
    printContentLine(" - You can take orders for 1000 tables (grid size)", width);

    snprintf(buffer, sizeof(buffer), " - But you only have %d chefs (SMs)", prop.multiProcessorCount);
    printContentLine(buffer, width);

    snprintf(buffer, sizeof(buffer), " - Each chef can cook %d dishes at once (blocks per SM)",
             prop.maxBlocksPerMultiProcessor);
    printContentLine(buffer, width);

    printContentLine(" - Orders wait in queue until a chef is free", width);
    printEmptyLine(width);
    printSectionFooter(width);
}

int main()
{
    const int BOX_WIDTH = 80;

    int deviceCount = 0;
    checkCudaErrors(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0)
    {
        std::cout << "No CUDA devices found" << std::endl;
        return 1;
    }

    std::cout << "Found " << deviceCount << " CUDA device(s)" << std::endl << std::endl;

    for (int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp deviceProp;
        checkCudaErrors(cudaGetDeviceProperties(&deviceProp, dev));

        std::cout << "Device " << dev << ": \"" << deviceProp.name << "\"" << std::endl << std::endl;

        // Print title
        printTitleBox("CUDA EXECUTION HIERARCHY", deviceProp.name, BOX_WIDTH);

        // Print device properties
        printDeviceProperties(deviceProp, BOX_WIDTH);

        // Print execution hierarchy
        printCudaHierarchy(deviceProp, BOX_WIDTH);

        if (dev < deviceCount - 1)
        {
            std::cout << std::string(BOX_WIDTH, '=') << std::endl << std::endl;
        }
    }

    return 0;
}