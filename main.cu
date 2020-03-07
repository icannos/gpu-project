/**************************************************************
Pierre Guetschel

***************************************************************/

#include <iostream>


// Genrates Gaussian distribution from a uniform one (Box-Muller)
__device__ void BoxMuller_d(float *g0, float *g1) {

    float loc;
    if (*g1 < 1.45e-6f) {
        loc = sqrtf(-2.0f * logf(0.00001f)) * cosf(*g0 * 2.0f * MoPI);
    } else {
        if (*g1 > 0.99999f) {
            loc = 0.0f;
        } else { loc = sqrtf(-2.0f * logf(*g1)) * cosf(*g0 * 2.0f * MoPI); }
    }
    *g0 = loc;
}

// Monte Carlo routine
__global__ void LDLt_max_k(int AGPU, int YGPU, int d) {
    int tidx = threadIdx.x % d;
    int Qt = (threadIdx.x - tidx) / d;
    int gbx = Qt + blockIdx.x * (blockDim.x / d);


    extern __shared__ float H[];

// Perform the LDLt factorization
    for (i = n; i > 0; i--) {
        if (tidx == 0) {
            for (k = n; k > i; k--) {
                sA[nt + n2 - i * (i + 1) / 2] -= sA[nt + n2 - k * (k + 1) / 2] *
                                                 sA[nt + n2 - k * (k + 1) / 2 + k - i] *
                                                 sA[nt + n2 - k * (k + 1) / 2 + k - i];
            }
        }
        __syncthreads();
        if (tidx < i - 1) {
            sA[nt + n2 - i * (i + 1) / 2 + tidx + 1] /= sA[nt + n2 - i * (i + 1) / 2];
            for (k = n; k > i; k--) {
                sA[nt + n2 - i * (i + 1) / 2 + tidx + 1] -= sA[nt + n2 - k * (k + 1) / 2] *
                                                            sA[nt + n2 - k * (k + 1) / 2 + k - i] *
                                                            sA[nt + n2 - k * (k + 1) / 2 + tidx + 1 + k - i] /
                                                            sA[nt + n2 - i * (i + 1) / 2];
            }
        }
        __syncthreads();
    }

}


int main() {
    float Tim;                            // GPU timer instructions
    cudaEvent_t start, stop;            // GPU timer instructions

    cudaMalloc(&res1, sizeof(float));
    cudaMemset(res2, 0.0f, sizeof(float));


    cudaEventCreate(&start);            // GPU timer instructions
    cudaEventCreate(&stop);                // GPU timer instructions
    cudaEventRecord(start, 0);            // GPU timer instructions

    LDLt_max_k << < NB, d * minTB, minTB * ((d * d + d) / 2 + d) * sizeof(float) >> > (AGPU, YGPU, d);

    cudaEventRecord(stop, 0);            // GPU timer instructions
    cudaEventSynchronize(stop);            // GPU timer instructions
    cudaEventElapsedTime(&Tim,            // GPU timer instructions
                         start, stop);                // GPU timer instructions
    cudaEventDestroy(start);            // GPU timer instructions
    cudaEventDestroy(stop);                // GPU timer instructions


    cudaMemcpy(&sum, res1, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(res1);


    printf("Execution time %f ms\n", Tim);

    return 0;
}
