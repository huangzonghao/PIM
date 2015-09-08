cmake_minimum_required ( VERSION 2.8)

find_package(CUDA)


# set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -dc)
# list(APPEND CUDA_NVCC_FLAGS "-dc")
# set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} –arch=sm_20)

file( GLOB_RECURSE  cuda_src  *.cu)
file( GLOB_RECURSE c_src *.cpp *.cc)

CUDA_ADD_EXECUTABLE(executable ${c_src} ${cuda_src})

