//
// Created by maxime on 08/03/20.
//

#ifndef CUDA_BASE_PARALLEL_SOLVER_H
#define CUDA_BASE_PARALLEL_SOLVER_H

#include "ide_params.h"

// Data description
// We store the data for a system into a 1-D array of the form

// [D_1 D_2 ... D_d T11 T21 T22 T31 T32 T33 ... Td1 Td2 ... Tdd]
// We denote D a pointer to the begining of that array
// and T a pointer to T11

// Each system takes (d + (d+1) / 2) * sizeof(float) bytes in memory
// A batch of N systems is then stored in an array of size N * (d + (d+1) / 2) * sizeof(float)

// Helpers

// Assuming you have a pointer T to T11, getL computes the indices of Lij in the array
// Therefore T[getL(i, j) = Lij
__device__ int getL(int i, int j);

// Same as for getL but for D, it is actually identity function
// Not used but here as an artefact from the past
__device__ int getD(int i);

// Parrallels ops

// Computes the sum of an array in O(log n) if there is O(n) thread, where n
// is the number of elements
__device__ void reduce_sum(float* T, int size);

// Invert a diagonal matrix
// ie just take the invert of the elements in parallel
__device__ void invert_diag(float * D, float* Y, int d);

// Solve a system whose matrix is triangular inferior
__device__ void solve_tinf(float* T, float* Y, int d);

// Solve a system whose matrix is triangular superior
__device__ void solve_tsup(float* T, float* Y, int d);

// Solve a system of the form LDLt X = Y
__device__ void solve_system(float* D, float* T, float* Y, int d);

// Take a batch of systems and solve one system by block.
__global__ void solve_batch(int N, int d, float* T, float* Y);

#endif //CUDA_BASE_PARALLEL_SOLVER_H
