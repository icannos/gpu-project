//
// Created by maxime on 26/03/20.
//

#ifndef CUDA_BASE_LDLT_CU_H
#define CUDA_BASE_LDLT_CU_H

int hgetLPierre(int d, int i, int j);
int hgetDPierre(int i);
void generate_systems(float *A, float *Y, int N, int d, bool verbose=true);
__device__ int getLPierre(int d, int i, int j);
__device__ int getDPierre(int d, int i);
__device__ void parallel_copy(float* dest, float* src, int n);
__global__ void LDLt_max_col_k(float* sA, int d);
__global__ void LDLt_max_row_k(float* sA, int d);
__global__ void LDLt_max_row_k_SHARED(float* sA, int d);

#endif //CUDA_BASE_LDLT_CU_H
