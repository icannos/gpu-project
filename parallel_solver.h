//
// Created by maxime on 08/03/20.
//

#ifndef CUDA_BASE_PARALLEL_SOLVER_H
#define CUDA_BASE_PARALLEL_SOLVER_H

#include "ide_params.h"

__device__ int getL(int d, int i, int j);
__device__ int getD(int i);
__device__ void solve_tinf(float* T, float* Y, int d);
__device__ void solve_tsup(float* T, float* Y, int d);
__device__ void prod_diag_trig(float * D, float* T, int d);
__device__ void solve_system(float* D, float* T, float* Y, int d);
__global__ void solve_batch(int N, int d, float* T, float* Y);

#endif //CUDA_BASE_PARALLEL_SOLVER_H
