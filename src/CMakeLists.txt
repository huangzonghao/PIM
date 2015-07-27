cmake_minimum_required ( VERSION 2.8)

find_package(CUDA)


file( GLOB_RECURSE  cuda_src  *.cu)
file( GLOB_RECURSE c_src *.cpp *.cc)

CUDA_ADD_EXECUTABLE(executable ${c_src} ${cuda_src})

