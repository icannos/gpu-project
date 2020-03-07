cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(CUDA_Test_Clion LANGUAGES CXX CUDA)
project(cuda_base CXX CUDA)

set(CMAKE_CXX_STANDARD 11)

add_library(cuda_base
        main.cu)
target_compile_features(cuda_base PUBLIC cxx_std_11)
set_target_properties( cuda_base
        PROPERTIES CUDA_SEPARABLE_COMPILATION ON
        )


set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_50,code=sm_50")

add_executable(CUDA_Test_Clion
        main.cu
        )

set_target_properties(
        CUDA_Test_Clion
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)
target_link_libraries(CUDA_Test_Clion cuda_base)