
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "ide_params.h"
#include "parallel_solver.h"

int hgetL(int d, int i, int j) {
    // If j > i, then we take the transpose of L
    if (j > i) {
        int t = i;
        i = j;
        j = t;
    }

    int l_position = i * (i - 1) / 2 + j;

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
            T[j] = ((float) (1));


        for (int j = 0; j<d; j++)
            T[hgetL(d, j,j)] = 1.0f;

        for (int j = 0; j < d; j++) {
            D[j] = (float) 1;
            Y[matrix_size * i + j] = ((float) (1));
        }

        if (verbose)
        {
            printf("[");
            for(int k = 0; k<d; k++)
            {
                printf("[");
                for(int j = 0; j<d; j++)
                {
                    if (j <= k)
                        printf("%f,", T[hgetL(d, j, k)]);
                    else
                        printf("%f,", 0.0f);
                }
                printf("],");
            }
            printf("]\n");
        }

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

void matrix_product(float *D, float *T, float *X, float *Y, int d) {
    for (int i = 0; i < d; i++) {
        Y[i] = 0;
        for (int k = i; k < d; k++) {
            Y[i] += T[hgetL(d, i, k)] * X[k];
        }

        Y[i] *= D[i];

    }

    memcpy(X, Y, sizeof(float) * d);

    for (int i = 0; i < d; i++) {
        Y[i] = 0;
        for (int k = 0; k <= i; k++) {
            Y[i] += T[hgetL(d, i, k)] * X[k];
        }
    }

}

int main(int argc, char *argv[]) {
    int d = 3;
    int n = 1;

    auto *A = (float *) malloc(sizeof(float) * n * (d + d * (d + 1) / 2));
    auto *Y = (float *) malloc(sizeof(float) * n * d);
    auto *Ychap = (float *) malloc(sizeof(float) * n * d);
    auto *X = (float *) malloc(sizeof(float) * n * d);

    float *gpuA;
    float *gpuY;

    cudaMalloc(&gpuA, sizeof(float) * n * (d + d * (d + 1) / 2));
    cudaMalloc(&gpuY, sizeof(float) * n * d);

    generate_systems(A, Y, n, d);


    cudaMemcpy(gpuA, A, sizeof(float) * n * (d + d * (d + 1) / 2), cudaMemcpyHostToDevice);
    cudaMemcpy(gpuY, Y, sizeof(float) * n * d, cudaMemcpyHostToDevice);

    solve_batch << < n, d >> > (n, d, gpuA, gpuY);

    cudaDeviceSynchronize();

    cudaMemcpy(X, gpuY, sizeof(float) * n * d, cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

    printf("[");
    for(int k = 0; k<d; k++)
        printf("%f,", X[k]);
    printf("]\n");
    printf("\0");


    cudaFree(gpuA);
    cudaFree(gpuY);
    free(A);
    free(Y);
    free(Ychap);
    free(X);

    return 0;
}
