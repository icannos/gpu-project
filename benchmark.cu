//
// Created by maxime on 26/03/20.
//

// Usage
// ./build/full N d

#include "LDLt.h"
#include "parallel_solver.h"

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <cmath>


int main(int argc, char* argv[]) {
    float Tim;                            // GPU timer instructions
    cudaEvent_t start, stop, startsolve, stopsolve;            // GPU timer instructions
    int d = 20;
    int n = 5;
    int num_thread_per_block = 1024;
    int factorizer = 0; // 0: columns || 1: rows || 2: shared memory+row

    n = atoi(argv[1]);
    d = atoi(argv[2]);
    num_thread_per_block = atoi(argv[3]);
    if (argc>4)
        factorizer = atoi(argv[4]);

    if (d>num_thread_per_block)
      throw std::invalid_argument( "d > num_thread_per_block" );
    if (factorizer==2 && d>64)
      throw std::invalid_argument( "d > 64: can not factorize big matrices on shared memory. Please use host memory" );
    num_thread_per_block = min(64, num_thread_per_block);

    // int minTB = 1;  // number of matrix per block
    int minTB = num_thread_per_block/d;  // number of matrix per block
    int NB = (n+minTB-1)/minTB;  // number of blocks (round up)

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
    printf("Execution time %f ms\n", Tim);  // GPU timer instructions


    cudaMemcpy(LandD, gpuA, sizeof(float) * n * (d + d * (d + 1) / 2), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    // cudaMemcpy(X, gpuY, sizeof(float) * n * d, cudaMemcpyDeviceToHost);
    // cudaDeviceSynchronize();

    Tim = 0;

    cudaEventCreate(&startsolve);                // GPU timer instructions
    cudaEventCreate(&stopsolve);                 // GPU timer instructions
    cudaEventRecord(startsolve, 0);              // GPU timer instructions

    solve_batch << < n, d, d* sizeof(float) >> > (n, d, gpuA, gpuY);

    cudaEventRecord(stopsolve, 0);               // GPU timer instructions
    cudaEventSynchronize(stopsolve);             // GPU timer instructions
    cudaEventElapsedTime(&Tim, startsolve, stopsolve);// GPU timer instructions
    cudaEventDestroy(startsolve);                // GPU timer instructions
    cudaEventDestroy(stopsolve);                 // GPU timer instructions

    printf("Solving time %f ms\n", Tim);  // GPU timer instructions




    cudaFree(gpuA);
    cudaFree(gpuY);
    free(A);
    free(LandD);
    free(Y);
    // free(Ychap);
    // free(X);

    return 0;
}
