//
// Created by maxime on 26/03/20.
//

#include "LDLt.h"
#include "parallel_solver.h"

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <cmath>

int main(int argc, char* argv[])
{
    float Tim;                            // GPU timer instructions
    cudaEvent_t start, stop;            // GPU timer instructions
    cudaEvent_t startsolve, stopsolve;

    int d = 100;
    int n = 100;

    // int num_thread_per_block = 1024;
    int num_thread_per_block = 300; // just to test
    int minTB = 1;  // number of matrix per block
    // int minTB = num_thread_per_block/d;  // number of matrix per block
    int NB = n/minTB;  // number of blocks
    int thread_number;

    n = atoi(argv[1]);
    d = atoi(argv[2]);

    if (d < 1024){
        thread_number = d;
    }
    else{
        thread_number = 1024; }


    srand(time(0));

    auto *A     = (float *) malloc(sizeof(float) * n * (d + d * (d + 1) / 2));
    auto *LandD = (float *) malloc(sizeof(float) * n * (d + d * (d + 1) / 2));
    auto *Y = (float *) malloc(sizeof(float) * n * d);

    auto *X = (float *) malloc(sizeof(float) * n * d);

    float *gpuA;
    float *gpuY;

    cudaMalloc(&gpuA, sizeof(float) * n * (d + d * (d + 1) / 2));
    cudaMalloc(&gpuY, sizeof(float) * n * d);

    //generate_systems(A, Y, n, d, false);

    cudaMemcpy(gpuA, A, sizeof(float) * n * (d + d * (d + 1) / 2), cudaMemcpyHostToDevice);


    cudaEventCreate(&start);                // GPU timer instructions
    cudaEventCreate(&stop);                 // GPU timer instructions
    cudaEventRecord(start, 0);              // GPU timer instructions

    //LDLt_max_col_k <<< NB, d * minTB, minTB * ((d * d + d) / 2 + d) * sizeof(float) >>> (gpuA, d);
    //LDLt_max_row_k <<< NB, d * minTB, minTB * ((d * d + d) / 2 + d) * sizeof(float) >>> (gpuA, d);
    cudaDeviceSynchronize();

    cudaEventRecord(stop, 0);               // GPU timer instructions
    cudaEventSynchronize(stop);             // GPU timer instructions
    cudaEventElapsedTime(&Tim, start, stop);// GPU timer instructions
    cudaEventDestroy(start);                // GPU timer instructions
    cudaEventDestroy(stop);                 // GPU timer instructions
    printf("Factorization time %f ms\n", Tim);  // GPU timer instructions


    cudaMemcpy(gpuY, Y, sizeof(float) * n * d, cudaMemcpyHostToDevice);
    Tim = 0;

    cudaEventCreate(&startsolve);                // GPU timer instructions
    cudaEventCreate(&stopsolve);                 // GPU timer instructions
    cudaEventRecord(startsolve, 0);              // GPU timer instructions

    solve_batch << < n, thread_number, thread_number* sizeof(float) >> > (n, d, gpuA, gpuY);

    cudaEventRecord(stopsolve, 0);               // GPU timer instructions
    cudaEventSynchronize(stopsolve);             // GPU timer instructions
    cudaEventElapsedTime(&Tim, startsolve, stopsolve);// GPU timer instructions
    cudaEventDestroy(startsolve);                // GPU timer instructions
    cudaEventDestroy(stopsolve);                 // GPU timer instructions

    printf("Solving time %f ms\n", Tim);  // GPU timer instructions

    //cudaMemcpy(LandD, gpuA, sizeof(float) * n * (d + d * (d + 1) / 2), cudaMemcpyDeviceToHost);
    //cudaDeviceSynchronize();

    cudaFree(gpuA);
    cudaFree(gpuY);
    free(X);
    free(A);
    free(LandD);
    free(Y);


    return 0;
}