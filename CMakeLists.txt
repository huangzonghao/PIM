# cmake_minimum_required(VERSION 2.8)

# project(Perishable_Inventory_Management)


# find_package(CUDA REQUIRED)

# if(CUDA_FOUND)
    # add_subdirectory(src)
    # include_directories(include)
    # INCLUDE_DIRECTORIES(/usr/local/cuda)
    # # set(CUDA_NVCC_FLAGS "-arch=compute_20; -code=sm_20; -rdc=true; -lcudadevrt")
    # set(CUDA_NVCC_FLAGS "-rdc=true; -lcudadevrt; -lcudart; -lcuda; -lmylib")

    # set( CMAKE_EXPORT_COMPILE_COMMANDS 1  )
# else()
    # message("ERROR: CUDA not found! please check the system configuration!")
# endif()

# set(CMAKE_CXX_FLAG "-g -Wall")

# message("\n"
        # "CMake Configuration for PIM\n"
        # "Instruction :\n"
        # "    1) make in this folder directly and the executable file can be found in ./src/excutable\n"
        # "    2) ./src/executable -h will print out the usage of the program\n"
        # "\n")


cmake_minimum_required(VERSION 2.8)

project(Perishable_Inventory_Management)
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda/")
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/CMake_Modules" ${CMAKE_MODULE_PATH})
find_package(CUDA)

set(CUDA_USE_STATIC_CUDA_RUNTIME ON)

set(CUDA_SEPARABLE_COMPILATION ON)
set(CUDA_BUILD_WITH_CUFFT_STATIC ON)
add_subdirectory(src)
include_directories(include)
