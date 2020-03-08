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

    int l_position = i*(i-1) / 2 + j;

    return l_position;
}

__device__ int getD(int i)
{
    int d_position = i;
    return d_position;
}

__device__ void solve_tinf(float* T, float* Y, int d)
{
    // Solve an equation of the form LZ = Y
    // T represents the inf triangular matrix
    // The results is then stored in Z

    for (int i = 0; i<d; i++)
    {
        __syncthreads();
        if (threadIdx.x < i)
        {
            Y[i] -= Y[threadIdx.x]*T[getL(d, i, threadIdx.x)];
        }
    }
}

__device__ void solve_tsup(float* T, float* Y, int d)
{
    // Solve an equation of the form LZ = Y
    // T represents the sup triangular matrix
    // The results is then stored in Z

    for (int i = 0; i<d; i++)
    {
        __syncthreads();
        if (threadIdx.x < i)
        {
            Y[d-i] -= Y[d-threadIdx.x]*T[getL(d, d-i, d-threadIdx.x)];
        }
    }
}

__device__ void prod_diag_trig(float * D, float* T, int d)
{
    if (threadIdx.x < d)
    {
        for (int i = 0; i<=d-threadIdx.x; i++)
        {
            T[getL(d, threadIdx.x, i)] *= D[threadIdx.x];
        }
    }
}

__device__ void solve_system(float* D, float* T, float* Y, int d)
{
    solve_tinf(T, Y, d);
    prod_diag_trig(D, T, d);
    solve_tsup(T, Y, d);
}

__global__ void solve_batch(int N, int d, float* T, float* Y)
{
    int matrix_size = d + d*(d+1) / 2;
    solve_system(&T[matrix_size*blockIdx.x], &T[matrix_size*blockIdx.x+d], &Y[matrix_size*blockIdx.x], d);
}