/**
 * Vector Addition
 *
 * This program demonstrates vector addition using C++.
 * It allocates memory on the host and performs the addition,
 * and measures the computation time for comparison with CUDA.
 *
 * Compilation: g++ vector_add.cpp -o cpp_vector_add.bin -O3
 * Execution: ./cpp_vector_add.bin
 */
#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <random>
using namespace std;

int* vector_add(const int* A, const int* B, int size) {
    int* C = new int[size];

    for (int i = 0; i < size; i++) {
        C[i] = A[i] + B[i];
    }
    return C;
}

int main() {
    // Use the same size as in CUDA version for comparison
    constexpr int N = 1 << 16;  // 65536 elements
    cout << "Vector size: " << N << endl;

    // Create and initialize vectors
    vector<int> a(N);
    vector<int> b(N);

    // Create a random generator lambda
    random_device rd;
    mt19937 gen(rd());
    auto random = [&gen]() { return gen() % 100; };

    // Fill with random values using the lambda
    generate(a.begin(), a.end(), random);
    generate(b.begin(), b.end(), random);

    // Allocate result array
    int* result = new int[N];

    // Measure only vector addition computation time
    auto start = chrono::high_resolution_clock::now();

    // Call vector_add function
    int* output = vector_add(a.data(), b.data(), N);

    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double, micro> computation_time = end - start;

    // Print timing information
    cout << "\nProfiling Results:" << endl;
    cout << "Vector addition time: " << computation_time.count() << " microseconds" << endl;

    // Clean up
    delete[] output;
    delete[] result;

    return 0;
}

/**
Vector size: 65536

Profiling Results:
Vector addition time: 209.778 microseconds

 */