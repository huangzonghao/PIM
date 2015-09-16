# set(source_code_dir "${PROJECT_SOURCE_DIR}/src")

# file( GLOB_RECURSE  cuda_src  "${source_code_dir}/*.cu")
# file( GLOB_RECURSE c_src "${source_code_dir}/*.cc")

# set(CUDA_NVCC_FLAGS "-arch=compute_20; -code=sm_20; -rdc=true; -lcudadevrt")

# CUDA_ADD_EXECUTABLE(executable ${c_src} ${cuda_src})




set(source_code_dir "${PROJECT_SOURCE_DIR}/src")

file( GLOB_RECURSE  cuda_src  "${source_code_dir}/*.cu")
file( GLOB_RECURSE c_src "${source_code_dir}/*.cc")


CUDA_ADD_EXECUTABLE(executable ${c_src} ${cuda_src})
