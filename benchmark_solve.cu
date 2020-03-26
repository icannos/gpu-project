
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>

#include "parallel_solver.h"

int hgetL(int d, int i, int j) {
    // If j > i, then we take the transpose of L
    if (j > i) {
        int t = i;
        i = j;
        j = t;
    }

    int l_position = i * (i - 1) / 2 + j-1;

    return l_position;
}

int hgetD(int i) {
    int d_position = i;
    return d_position;
}

void generate_systems(float *A, float *Y, int N, int d, bool verbose=true) {
    int matrix_size = d + d * (d + 1) / 2;

    for (int i = 0; i < N; i++) {
        float *D = &A[i * matrix_size];
        float *T = &A[i * matrix_size + d];


        for (int j = 0; j < (d * (d + 1) / 2); j++)
            T[j] = ((float) (1+rand()%2));

        for (int j = 1; j<=d; j++)
            T[hgetL(d, j, j)] = 1.0f;

        for (int j = 0; j < d; j++) {
            D[j] = (float) (1+rand()%5);
            Y[d * i + j] = ((float) (1+rand()%2));
        }

        if (verbose)
        {
            printf("[");
            for(int k = 1; k<=d; k++)
            {
                printf("[");
                for(int j = 1; j<=d; j++)
                {
                    if (j <= k)
                        printf("%f,", T[hgetL(d, j, k)]);
                    else
                        printf("%f,", 0.0f);
                }
                printf("],");
            }
            printf("]\n");

            printf("[");
            for(int k = 0; k<d; k++)
                printf("%f,", D[k]);
            printf("]\n");

            printf("[");
            for(int k = 0; k<d; k++)
                printf("%f,", Y[k]);
            printf("]\n");
        }



    }
}

int main(int argc, char *argv[]) {
    int d = 150;
    int n = 9;

    float Tim;                            // GPU timer instructions
    cudaEvent_t start, stop;            // GPU timer instructions
    cudaEvent_t startsolve, stopsolve;

    int thread_number = 1024;

    n = atoi(argv[1]);
    d = atoi(argv[2]);
    thread_number = atoi(argv[3]);

    srand(time(0));

    auto *A = (float *) malloc(sizeof(float) * n * (d + d * (d + 1) / 2));
    auto *Y = (float *) malloc(sizeof(float) * n * d);
    auto *Ychap = (float *) malloc(sizeof(float) * n * d);
    auto *X = (float *) malloc(sizeof(float) * n * d);

    float *gpuA;
    float *gpuY;

    cudaMalloc(&gpuA, sizeof(float) * n * (d + d * (d + 1) / 2));
    cudaMalloc(&gpuY, sizeof(float) * n * d);

    generate_systems(A, Y, n, d, false);


    cudaMemcpy(gpuA, A, sizeof(float) * n * (d + d * (d + 1) / 2), cudaMemcpyHostToDevice);
    cudaMemcpy(gpuY, Y, sizeof(float) * n * d, cudaMemcpyHostToDevice);

    cudaEventCreate(&start);                // GPU timer instructions
    cudaEventCreate(&stop);                 // GPU timer instructions
    cudaEventRecord(start, 0);              // GPU timer instructions

    solve_batch << < n, thread_number, thread_number* sizeof(float) >> > (n, d, gpuA, gpuY);

    cudaDeviceSynchronize();

    cudaEventRecord(stop, 0);               // GPU timer instructions
    cudaEventSynchronize(stop);             // GPU timer instructions
    cudaEventElapsedTime(&Tim, start, stop);// GPU timer instructions
    cudaEventDestroy(start);                // GPU timer instructions
    cudaEventDestroy(stop);                 // GPU timer instructions
    printf("Solving time %f ms\n", Tim);  // GPU timer instructions

    cudaMemcpy(X, gpuY, sizeof(float) * n * d, cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

/*    printf("[");
    for(int k = 0; k<d; k++)
        printf("%f,", X[k]);
    printf("]\n");
    printf("\0");*/


    cudaFree(gpuA);
    cudaFree(gpuY);
    free(A);
    free(Y);
    free(Ychap);
    free(X);

    return 0;
}
