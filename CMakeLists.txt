cmake_minimum_required ( VERSION 2.8)

project(Perishable_Inventory_Management)
find_package(CUDA)

# set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} –arch=sm_20)
# i don't even know how did the following setting work...
# http://codedmi.com/questions/768126/using-cuda-object-linking-with-cmake
set(CUDA_NVCC_FLAGS "-arch=compute_20; -code=sm_20; -rdc=true; -lcudadevrt")
# list(APPEND CUDA_NVCC_FLAGS "-dc")

# include_directories(BEFORE PROJECT_SOURCE_DIR)
include_directories(PROJECT_SOURCE_DIR/include)
add_subdirectory(src)
set( CMAKE_EXPORT_COMPILE_COMMANDS 1  )

