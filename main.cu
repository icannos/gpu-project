/**************************************************************
Pierre Guetschel

***************************************************************/

#include <iostream>


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
            T[hgetL(d, j,j)] = 1.0f;

        for (int j = 0; j < d; j++) {
            D[j] = (float) (1+rand()%5);
            Y[matrix_size * i + j] = ((float) (1+rand()%2));
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

// ************************************************************************ //


__global__ void LDLt_max_k(int AGPU, int d) {
    int tidx = threadIdx.x % d;
    int Qt = (threadIdx.x - tidx) / d;
    int gbx = Qt + blockIdx.x * (blockDim.x / d);


    extern __shared__ float H[];

// Perform the LDLt factorization
    for (i = n; i > 0; i--) {
        if (tidx == 0) {
            for (k = n; k > i; k--) {
                sA[nt + n2 - i * (i + 1) / 2] -=
                  sA[nt + n2 - k * (k + 1) / 2] *
                  sA[nt + n2 - k * (k + 1) / 2 + k - i] *
                  sA[nt + n2 - k * (k + 1) / 2 + k - i];
            }
        }
        __syncthreads();
        if (tidx < i - 1) {
            sA[nt + n2 - i * (i + 1) / 2 + tidx + 1] /= sA[nt + n2 - i * (i + 1) / 2];
            for (k = n; k > i; k--) {
                sA[nt + n2 - i * (i + 1) / 2 + tidx + 1] -=
                  sA[nt + n2 - k * (k + 1) / 2] *
                  sA[nt + n2 - k * (k + 1) / 2 + k - i] *
                  sA[nt + n2 - k * (k + 1) / 2 + tidx + 1 + k - i] /
                  sA[nt + n2 - i * (i + 1) / 2];
            }
        }
        __syncthreads();
    }

}

// ************************************************************************ //


int main() {
    float Tim;                            // GPU timer instructions
    cudaEvent_t start, stop;            // GPU timer instructions
    int d = 11;
    int n = 1;

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

    generate_systems(A, Y, n, d);

    cudaMemcpy(gpuA, A, sizeof(float) * n * (d + d * (d + 1) / 2), cudaMemcpyHostToDevice);
    // cudaMemcpy(gpuY, Y, sizeof(float) * n * d, cudaMemcpyHostToDevice);



    cudaEventCreate(&start);                // GPU timer instructions
    cudaEventCreate(&stop);                 // GPU timer instructions
    cudaEventRecord(start, 0);              // GPU timer instructions

    LDLt_max_k << < NB, d * minTB, minTB * ((d * d + d) / 2 + d) * sizeof(float) >> > (gpuA, d);
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



    // printf("[");
    // for(int k = 0; k<d; k++)
    //     printf("%f,", X[k]);
    // printf("]\n");
    // printf("\0");

    // print A
    printf("This is A :\n");
    printf("[");
    for(int i = 0; i<d; i++){
      printf("[");
      for(int k = 0; j<d; j++){
        if(i==j)
          printf("%f,",A[hgetD(i)]);
        else
          printf("%f,",A[hgetL(d,i,j)]);
      }
      printf("],");
    }
    printf("]\n");
    printf("\0");

    // print L
    printf("This is L :\n");
    printf("[");
    for(int i = 0; i<d; i++){
      printf("[");
      for(int k = 0; j<d; j++){
        if(j>i)
          printf("%f,",0.0f);
        else
          printf("%f,",LandD[hgetL(d,i,j)]);
      }
      printf("],");
    }
    printf("]\n");
    printf("\0");

    // print D
    printf("This is D :\n");
    printf("[");
    for(int i = 0; i<d; i++)
      printf("%f,",LandD[hgetD(i)]);
    printf("]\n");
    printf("\0");



    cudaFree(gpuA);
    cudaFree(gpuY);
    free(A);
    free(LandD);
    free(Y);
    // free(Ychap);
    // free(X);

    return 0;
}
