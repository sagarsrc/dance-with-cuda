/**
 * Matrix Multiplication
 *
 * This program demonstrates square matrix multiplication using C++.
 * It allocates memory on the host and performs the multiplication,
 * and measures the computation time for comparison with CUDA.
 *
 * Compilation: g++ matmul.cpp -o cpp_matmul.bin -O3
 * Execution: ./cpp_matmul.bin
 */
#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <random>
using namespace std;

void initialize_matrix(vector<vector<int>> &matrix, int size)
{
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            matrix[i][j] = rand() % 10;
        }
    }
}

void matrix_multiplication(const vector<vector<int>> &A, const vector<vector<int>> &B, vector<vector<int>> &C, int size)
{
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            C[i][j] = 0;
            for (int k = 0; k < size; k++)
            {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

int main()
{
    int size = 1024;
    vector<vector<int>> A(size, vector<int>(size));
    vector<vector<int>> B(size, vector<int>(size));
    vector<vector<int>> C(size, vector<int>(size));

    initialize_matrix(A, size);
    initialize_matrix(B, size);

    auto start = chrono::high_resolution_clock::now();
    matrix_multiplication(A, B, C, size);
    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);

    cout << "Time taken: " << duration.count() << " milliseconds" << endl;

    if (false)
    {
        cout << "Matrix A:" << endl;
        for (int i = 0; i < size; i++)
        {
            for (int j = 0; j < size; j++)
            {
                cout << A[i][j] << " ";
            }
            cout << endl;
        }

        cout << "Matrix B:" << endl;
        for (int i = 0; i < size; i++)
        {
            for (int j = 0; j < size; j++)
            {
                cout << B[i][j] << " ";
            }
            cout << endl;
        }

        cout << "Matrix C:" << endl;
        for (int i = 0; i < size; i++)
        {
            for (int j = 0; j < size; j++)
            {
                cout << C[i][j] << " ";
            }
            cout << endl;
        }
    }
    return 0;
}