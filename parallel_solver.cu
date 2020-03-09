//
// Created by maxime on 08/03/20.
//

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "ide_params.h"
#include "parallel_solver.h"


__device__ int getL(int d, int i, int j)
{
    // If j > i, then we take the transpose of L
    if (j > i) {int t = i; i = j; j = t;}

    int l_position = i*(i-1) / 2 + j-1;

    return l_position;
}

__device__ int getD(int i)
{
    int d_position = i;
    return d_position;
}

__device__ void reduce_sum(float* T, int size)
{
    if (size > 1) {
        for (unsigned int s = 1; s < size; s *= 2) {
            if (threadIdx.x % (2 * s) == 0) {
                T[threadIdx.x] += T[threadIdx.x + s];
            }

            __syncthreads();
        }
    }
}

__device__ void solve_tinf(float* T, float* Y, int d)
{
    // Solve an equation of the form LZ = Y
    // T represents the inf triangular matrix
    // The results is then stored in Z

    extern __shared__ float tmp[];

    for (int i = 1; i<=d; i++)
    {
        __syncthreads();
        if (threadIdx.x < i)
        {
            tmp[threadIdx.x] =  Y[threadIdx.x]*T[getL(d, i+1, threadIdx.x+1)];
            reduce_sum(tmp, i);

            __syncthreads();
            if (threadIdx.x == 0)
                Y[i] -= tmp[0];
        }
    }
}

__device__ void solve_tsup(float* T, float* Y, int d)
{
    // Solve an equation of the form LZ = Y
    // T represents the sup triangular matrix
    // The results is then stored in Z

    extern __shared__ float tmp[];

    for (int i = 1; i<=d; i++)
    {
        __syncthreads();
        if (threadIdx.x < i)
        {
            tmp[threadIdx.x] = Y[d-threadIdx.x]*T[getL(d, d-i+1, d-threadIdx.x+1)];

            reduce_sum(tmp, i);

            __syncthreads();
            if (threadIdx.x == 0)
                Y[d-i] -= tmp[0];
        }
    }
}

__device__ void invert_diag(float * D, float* Y, int d)
{
    if (threadIdx.x < d)
        Y[threadIdx.x] /= D[threadIdx.x];
}

__device__ void solve_system(float* D, float* T, float* Y, int d)
{
    //solve_tinf(T, Y, d);
    //invert_diag(D, Y, d);
    solve_tsup(T, Y, d);
}

__global__ void solve_batch(int N, int d, float* T, float* Y)
{
    int matrix_size = d + d*(d+1) / 2;
    solve_system(&T[matrix_size*blockIdx.x], &T[matrix_size*blockIdx.x+d], &Y[matrix_size*blockIdx.x], d);
}