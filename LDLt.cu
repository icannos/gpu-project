# include "utils.h"



int hgetL(int d, int i, int j) {
    // If j > i, then we take the transpose of L
    if (j > i) {
        int t = i;
        i = j;
        j = t;
    }

    int l_position    =     i*(i+1) / 2 + j;
    // int l_position =     i*(i-1) / 2 + j-1;

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


        for (int j=0; j < (d * (d + 1) / 2); j++)
            T[j] = ((float) (1+rand()%2));


        for (int j=0; j<d; j++)
            T[hgetL(d, j,j)] = 1.0f;

        for (int j=0; j<d; j++) {
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

// ************************************************************************ //

// __device__ int getL(float* T, int n, int d, int matrix_id, int i, int j)
__device__ int getL(int d, int i, int j)
{
    // If j > i, then we take the transpose of L
    if (j > i) {int t = i; i = j; j = t;};

    // int matrix_memory_size = (d+d*(d+1)/2);
    int l_position    = d + i*(i+1) / 2 + j;
    // int l_position =     i*(i-1) / 2 + j-1;
    // int l_position = d + i*(i-1) / 2 + j;
    return l_position;
    // return &T[matrix_id * matrix_memory_size + l_position]
}

// __device__ int getD(float* T, int n, int d, int matrix_id, int i)
__device__ int getD(int d, int i)
{
    // int matrix_memory_size = (d+d*(d+1)/2);
    int d_position = i;
    return d_position;
    // return &T[matrix_id * matrix_memory_size + d_position]
}

__device__ void parallel_copy(float* dest, float* src, int n)
{
  int i = threadIdx.x;
  int stride = blockDim.x;
  while(i<n){
    dest[i] = src[i];
    i += stride;
  }
  __syncthreads();
}

// __global__ void LDLt_max_col_k(float* AGPU, int d)
__global__ void LDLt_max_col_k(float* sA, int d)
{
  int tidx = threadIdx.x%d;
  int Qt = (threadIdx.x-tidx)/d;
  int A_size = d*(d+1)/2+d;
  int minTB = blockDim.x/d;
  // printf("minTB %d\n", minTB);
  int nt = (blockIdx.x*minTB + Qt) * A_size;
  // int gbx = Qt + blockIdx.x*(blockDim.x/d);


  // extern __shared__ float sA[];
  // //copy ACPU to sA
  // parallel_copy(sA, &AGPU[(blockIdx.x*minTB + Qt)*A_size], minTB*A_size);

  // tidx==i

  // Perform the LDLt factorization
  int j, k;
  for(j=0; j<d; j++){
    // D_j,j :
    if(tidx==0){
      for(k=0; k<j; k++){
        sA[nt+getD(d, j)] -= sA[nt+getD(d,k)]*
          sA[nt+getL(d,j,k)]*
          sA[nt+getL(d,j,k)];
      }
    }
    __syncthreads();

    // L_:,j parallel
    if(tidx>j){
      printf("(%d,%d,%d,%d),", nt+getL(d,tidx,j), nt, tidx, j);
      sA[nt+getL(d,tidx,j)] /= sA[nt+getD(d,j)];
      for(k=0; k<j; k++){
        sA[nt+getL(d,tidx,j)] -= sA[nt+getL(d,tidx,k)]*
          sA[nt+getL(d,j,k)]*
          sA[nt+getD(d,k)]/
          sA[nt+getD(d,j)];
      }
    }
    __syncthreads();
  }

  // parallel_copy(&AGPU[(blockIdx.x*minTB + Qt)*A_size], sA, minTB*A_size);

}

// __global__ void LDLt_max_row_k(float* AGPU, int d)
__global__ void LDLt_max_row_k(float* sA, int d)
{
  int tidx = threadIdx.x%d;
  int Qt = (threadIdx.x-tidx)/d;
  int A_size = d*(d+1)/2+d;
  int minTB = blockDim.x/d;
  // printf("minTB %d\n", minTB);
  int nt = (blockIdx.x*minTB + Qt) * A_size;
  // int gbx = Qt + blockIdx.x*(blockDim.x/d);


  // extern __shared__ float sA[];
  // //copy ACPU to sA
  // parallel_copy(sA, &AGPU[(blockIdx.x*minTB + Qt)*A_size], minTB*A_size);

  // tidx==j

  // Perform the LDLt factorization
  int i, k;
  for(i=0; i<d; i++){
    // D_i,i :
    if(tidx==0){
      for(k=0; k<i; k++){
        sA[nt+getD(d, i)] -= sA[nt+getD(d,k)]*
          sA[nt+getL(d,i,k)]*
          sA[nt+getL(d,i,k)];
      }
    }
    __syncthreads();

    // L_i,: parallel
    if(i<tidx){
      printf("(%d,%d,%d,%d),", nt+getL(d,i,tidx), nt, i,tidx);
      sA[nt+getL(d,i,tidx)] /= sA[nt+getD(d,i)];
      for(k=0; k<i; k++){
        sA[nt+getL(d,i,tidx)] -= sA[nt+getL(d,k,tidx)]*
          sA[nt+getL(d,k,i)]*
          sA[nt+getD(d,k)]/
          sA[nt+getD(d,i)];
      }
    }
    __syncthreads();
  }

  // parallel_copy(&AGPU[(blockIdx.x*minTB + Qt)*A_size], sA, minTB*A_size);

}



// ************************************************************************ //


int main() {
    float Tim;                            // GPU timer instructions
    cudaEvent_t start, stop;            // GPU timer instructions
    int d = 20;
    int n = 1;
    // int num_thread_per_block = 1024;
    int num_thread_per_block = 300; // just to test
    int minTB = 1;  // number of matrix per block
    // int minTB = num_thread_per_block/d;  // number of matrix per block
    int NB = n/minTB;  // number of blocks

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
    LDLt_max_row_k <<< NB, d * minTB, minTB * ((d * d + d) / 2 + d) * sizeof(float) >>> (gpuA, d);
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



    // printf("[");
    // for(int k = 0; k<d; k++)
    //     printf("%f,", X[k]);
    // printf("]\n");
    // printf("\0");

    printf("{\n");
    // print A
    printf("'A':[");
    for(int i = 0; i<d; i++){
      printf("[");
      for(int j = 0; j<d; j++){
        if(i==j)
          printf("%f,",A[hgetD(i)]);
        else
          printf("%f,",A[d+hgetL(d,i,j)]);
      }
      printf("],");
    }
    printf("],\n");
    // printf("\0");

    // print L
    printf("'L':[");
    for(int i = 0; i<d; i++){
      printf("[");
      for(int j = 0; j<d; j++){
        if(j>i)
          printf("%f,",0.0f);
        else
          printf("%f,",LandD[d+hgetL(d,i,j)]);
      }
      printf("],");
    }
    printf("],\n");
    // printf("\0");

    // print D
    printf("'D':[");
    for(int i = 0; i<d; i++)
      printf("%f,",LandD[hgetD(i)]);
    printf("],\n");
    // printf("\0");

    // print ones
    printf("'ones':[");
    for(int i = 0; i<d; i++)
      printf("%f,",A[d+hgetL(d, i, i)]);
    printf("],\n");
    // printf("\0");

    printf("}\n");


    cudaFree(gpuA);
    cudaFree(gpuY);
    free(A);
    free(LandD);
    free(Y);
    // free(Ychap);
    // free(X);

    return 0;
}
