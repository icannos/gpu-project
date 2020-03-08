//
// Created by maxime on 08/03/20.
//

#ifndef CUDA_BASE_IDE_PARAMS_H
#define CUDA_BASE_IDE_PARAMS_H

//#define __JETBRAINS_IDE__

#ifdef __JETBRAINS_IDE__
#define __CUDACC__ 1
#define __Host__
#define __device__
#define __global__
#define __forceinline__
#define __shared__
#define cudaMemcpyHostToDevice 0
#define cudaMemcpyDeviceToHost 1
inline void __syncthreads() {}
inline void cudaMalloc(void*, size_t) {}
inline void cudaDeviceSynchronize() {}
inline void cudaMemcpy(void*, void*, size_t, int);
inline void __threadfence_block() {}
template<class T> inline T __clz(const T val) { return val; }
struct __cuda_fake_struct { int x; };
extern __cuda_fake_struct blockDim;
extern __cuda_fake_struct threadIdx;
extern __cuda_fake_struct blockIdx;
#endif

#endif //CUDA_BASE_IDE_PARAMS_H
