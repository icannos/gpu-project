//
// Created by maxime on 08/03/20.
//

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "ide_params.h"
#include "parallel_solver.h"

// Helpers
__device__ int getL(int i, int j) {

    // If j > i, then we take the transpose of L
    if (j > i) {
        int t = i;
        i = j;
        j = t;
    }

    // Computes the position in T from i j
    // Lij is at place (i * (i-1) / 2 + j) -1
    // There are minus one because we start from 1 and not from 0
    // getL(1, 1) is L11
    int l_position = i * (i - 1) / 2 + j - 1;

    return l_position;
}

__device__ int getD(int i) {
    int d_position = i;
    return d_position;
}

// Usual parallel sum of an array
// Assuming we get an array begining at T and of size size.
__device__ void reduce_sum(float *T, int size) {
    __syncthreads();
    if (size > 1) {
        for (unsigned int s = 1; s < size; s *= 2) {
            __syncthreads();
            if (threadIdx.x % (2 * s) == 0) {
                T[threadIdx.x] += T[threadIdx.x + s];
            }
        }
    }
}


// Assuming that T is the begining of the triangular matrix
// Y the target
// d the dimension of the problem
__device__ void solve_tinf(float *T, float *Y, int d) {
    // Solve an equation of the form LZ = Y
    // T represents the inf triangular matrix
    // The results are computed in place

    /*
        We use the following formula
        x_1 = y_1
        x_2 = y_2 - L21 x_1
        x_3 = y_3 - L31 x_1 - L32 x_2
        ...

        We compute each Lij x_j in parallel and reduce the sum in parallel
        It is cut if we dont have enough thread
     */

    extern __shared__ float tmp[];

    int blockdim = blockDim.x;
    int threadid = threadIdx.x;

    for (int i = 1; i < d; i++) {
        // blockdim is the number of thread we have
        // q is the number of cut we have to do
        int q = i / blockdim;
        // and if there is a remainder
        int rmd = i % blockdim;

        // general case
        __syncthreads();
        for (int k = 0; k < q; k++) {

            if (threadid < blockdim) {
                // We compute Lij x_j
                // And we store it in shared memory for further processing
                tmp[threadid] = Y[k * blockdim + threadid] * T[getL(i + 1, k * blockdim + threadid + 1)];

                // Then we compute the reduced sum
                reduce_sum(tmp, blockdim);
                __syncthreads();

                // And we substract it from the result
                // What we do is actually
                // Y_i -= (first cut) + (second cut) ...
                if (threadid == 0)
                    Y[i] -= tmp[0];
                tmp[threadid] = 0;

            }
            __syncthreads();
        }

        // Remainder
        // Same as before but not using all the thread.
        __syncthreads();
        if (threadid < rmd) {
            tmp[threadid] = Y[q * blockdim + threadid] * T[getL(i + 1, q * blockdim + threadid + 1)];

            __syncthreads();
            reduce_sum(tmp, rmd);

            __syncthreads();
            if (threadid == 0)
                Y[i] -= tmp[0];
        }
        tmp[threadid] = 0;
        __syncthreads();
    }
}

__device__ void solve_tsup(float *T, float *Y, int d) {
    // Solve an equation of the form LZ = Y
    // T represents the sup triangular matrix
    // The results is then stored in Y

    // Same method than the previous, just we go bottom up instead
    extern __shared__ float tmp[];

    int blockdim = blockDim.x;
    int threadid = threadIdx.x;

    for (int i = 1; i <= d; i++) {

        int q = i / blockdim;
        int rmd = i % blockdim;

        for (int k = 0; k < q; k++) {
            __syncthreads();
            if (threadid < blockdim) {

                __syncthreads();
                if (threadIdx.x < blockdim) {
                    tmp[threadid] = Y[d - (k*blockdim+threadid)] * T[getL(d - i + 1, d - (k * blockdim + threadid) + 1)];

                    reduce_sum(tmp, blockdim);

                    __syncthreads();
                    if (threadIdx.x == 0)
                        Y[d - i] -= tmp[0];
                }
            }
            tmp[threadid] = 0;
        }

        if (threadid < rmd) {

            __syncthreads();
            if (threadIdx.x < rmd) {
                tmp[threadid] = Y[d - (q*blockdim+threadid)] * T[getL(d - i + 1, d - (q * blockdim + threadid) + 1)];

                reduce_sum(tmp, rmd);

                __syncthreads();
                if (threadIdx.x == 0)
                    Y[d - i] -= tmp[0];
            }
            tmp[threadid] = 0;
        }
    }
}

__device__ void invert_diag(float *D, float *Y, int d) {
    // Invert a diagonal matrix in parallel
    // It is just each element 1/x

    int q = d / blockDim.x;
    int rmd = d % blockDim.x;

    for (int k = 0; k < q; k++) {
        if (threadIdx.x < blockDim.x)
            Y[k * blockDim.x + threadIdx.x] /= D[k * blockDim.x + threadIdx.x];
    }

    if (threadIdx.x < rmd)
        Y[q * blockDim.x + threadIdx.x] /= D[q * blockDim.x + threadIdx.x];

}

// Solve a system in the LDLt form
// The result is stored into Y
__device__ void solve_system(float *D, float *T, float *Y, int d) {
    solve_tinf(T, Y, d);
    invert_diag(D, Y, d);
    solve_tsup(T, Y, d);
}

// Encapsulation of the solver, each block take care of one matrix
__global__ void solve_batch(int N, int d, float *T, float *Y) {
    int matrix_size = d + d * (d + 1) / 2;
    solve_system(&T[matrix_size * blockIdx.x], &T[matrix_size * blockIdx.x + d], &Y[d * blockIdx.x], d);
}