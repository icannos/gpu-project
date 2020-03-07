
#include <iostream>

__global__ void solve_batch(int N, int d, float* T, float* Y)
{

}

__device__ void solve(float* T, float* Y, int n, int d, int matrix_id)
{

}

__device__ weighte_sum(float* Z, float* T, int number)
{
  for(i=n; i > 0; i/=2)
  {
      if threadIdx.x
  }
}

__device__ float* getL(float* T, int n, int d, int matrix_id, int i, int j)
{
    // If j > i, then we take the transpose of L
    if (j > i) {int t = i; i = j; j = t}

    int matrix_memory_size = (d+d*(d+1)/2)
    int l_position = d + i*(i-1) / 2 + j

    return &T[matrix_id * matrix_memory_size + l_position]
}

__device__ float* getD(float* T, int n, int d, int matrix_id, int i)
{
    int matrix_memory_size = (d+d*(d+1)/2)
    int d_position = i

    return &T[matrix_id * matrix_memory_size + d_position]
}

int main(int *argc, char*[] argv)
{

}
