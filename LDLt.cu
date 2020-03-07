# include "utils.h"


// __device__ int getL(float* T, int n, int d, int matrix_id, int i, int j)
__device__ int getL(int d, int i, int j)
{
    // If j > i, then we take the transpose of L
    if (j > i); {int t = i; i = j; j = t;};

    // int matrix_memory_size = (d+d*(d+1)/2);
    int l_position = d + i*(i-1) / 2 + j;
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

__global__ void LDLt_max_col_k_shared(float* ACPU, int d)
{
  int tidx = threadIdx.x%d;
  int Qt = (threadIdx.x-tidx)/d;
  int A_size = (d*d+d)/2+d;
  int minTB = blockDim.x/d;
  int nt = Qt * A_size;
  // int gbx = Qt + blockIdx.x*(blockDim.x/d);


  extern __shared__ float sA[];
  //copy ACPU to sA
  parallel_copy(sA, &ACPU[(blockIdx.x*minTB + Qt)*A_size], minTB*A_size);



  // Perform the LDLt factorization
  int j, k;
  for(j=0; j<d; j++){ // i  == j in paper
    // D_j,j :
    if(tidx==0){ // tidx==i
      for(k=0; k<j; k++){
        sA[nt+getD(d, j)] -= sA[nt+getD(d,k)]*
          sA[nt+getL(d,j,k)]*
          sA[nt+getL(d,j,k)];
      }
    }
    __syncthreads();

    // L_:,j parallel
    if(tidx>j){
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

  parallel_copy(&ACPU[(blockIdx.x*minTB + Qt)*A_size], sA, minTB*A_size);

}


int main(int *argc, char*[] argv)
{
  int n = 5;
  int d = 6;
  // int num_thread_per_block = 1024;
  int num_thread_per_block = 15; // just to test
  int minTB = num_thread_per_block/d;
  int NB = n/minTB;

  float* A;
  int A_size = (d*d+d)/2+d;



  cudaMalloc(&A, sizeof(float)*n*A_size);
  init_A(A,n,d);
  print("A[0] before : %f", A[0])

  <<< NB, d*minTB, minTB*A_size*sizeof(float)>>>(A, d);

  print("A[0] after : %f", A[0])

}
