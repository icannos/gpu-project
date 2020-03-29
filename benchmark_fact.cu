//
// Pierre G 27/03/20.
//


// Usage
// ./build/fact N d num_thread_per_block
// ./build/fact N d num_thread_per_block  |  python verify_facto.py --atol 0.01


#include "LDLt.h"

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <cmath>


int main(int argc, char* argv[]) {
    float Tim;                            // GPU timer instructions
    cudaEvent_t start, stop;            // GPU timer instructions
    int d = 20;
    int n = 5;
    int num_thread_per_block = 1024;
    int factorizer = 0; // 0: columns || 1: rows || 2: shared memory+row

    n = atoi(argv[1]); // Number of matrices
    d = atoi(argv[2]); // Number of dimension
    num_thread_per_block = atoi(argv[3]);
    if (argc>4)
        factorizer = atoi(argv[4]);

    // int minTB = 1;  // number of matrix per block
    int minTB = num_thread_per_block/d;  // number of matrix per block
    int NB = (n+minTB-1)/minTB;  // number of blocks (round up)
    printf("%d %d", minTB, NB);

    srand(time(0));

    auto *A     = (float *) malloc(sizeof(float) * n * (d + d * (d + 1) / 2));
    auto *LandD = (float *) malloc(sizeof(float) * n * (d + d * (d + 1) / 2));
    auto *Y = (float *) malloc(sizeof(float) * n * d);
    // auto *Ychap = (float *) malloc(sizeof(float) * n * d);
    auto *X = (float *) malloc(sizeof(float) * n * d);

    float *gpuA;
    float *gpuY;

    cudaMalloc(&gpuA, sizeof(float) * n * (d + d * (d + 1) / 2));
    cudaMalloc(&gpuY, sizeof(float) * n * d);

    generate_systems(A, Y, n, d, false);

    cudaMemcpy(gpuA, A, sizeof(float) * n * (d + d * (d + 1) / 2), cudaMemcpyHostToDevice);
    // cudaMemcpy(gpuY, Y, sizeof(float) * n * d, cudaMemcpyHostToDevice);



    cudaEventCreate(&start);                // GPU timer instructions
    cudaEventCreate(&stop);                 // GPU timer instructions
    cudaEventRecord(start, 0);              // GPU timer instructions

    // LDLt_max_col_k <<< NB, d * minTB, minTB * ((d * d + d) / 2 + d) * sizeof(float) >>> (gpuA, d);
    // LDLt_max_row_k <<< NB, d * minTB, minTB * ((d * d + d) / 2 + d) * sizeof(float) >>> (gpuA, d);
    if (factorizer==0)
        LDLt_max_col_k <<< NB, d * minTB, 0 >>> (gpuA, d);
    else if (factorizer==1)
        LDLt_max_row_k <<< NB, d * minTB, 0 >>> (gpuA, d);
    else if (factorizer==2)
        LDLt_max_row_k_SHARED <<< NB, d * minTB, minTB * ((d * d + d) / 2 + d) * sizeof(float) >>> (gpuA, d);
    else
        throw std::invalid_argument( "unknown factorizer" );
    cudaDeviceSynchronize();

    cudaEventRecord(stop, 0);               // GPU timer instructions
    cudaEventSynchronize(stop);             // GPU timer instructions
    cudaEventElapsedTime(&Tim, start, stop);// GPU timer instructions
    cudaEventDestroy(start);                // GPU timer instructions
    cudaEventDestroy(stop);                 // GPU timer instructions
    printf("\nExecution time %f ms\n", Tim);  // GPU timer instructions


    cudaMemcpy(LandD, gpuA, sizeof(float) * n * (d + d * (d + 1) / 2), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    // cudaMemcpy(X, gpuY, sizeof(float) * n * d, cudaMemcpyDeviceToHost);
    // cudaDeviceSynchronize();

    return 0;
}
