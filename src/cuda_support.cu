/*
 * =============================================================================
 *
 *       Filename:  cuda_support.cu
 *
 *    Description:  This file contains the cuda functions which provides  general
 *                    support of the task and have nothing to do with the
 *                    specific algorithm and calculation
 *                    Mostly the interface for other cpp source file
 *
 *        Created:  Thu Jul 23 03:38:40 2015
 *       Modified:  Tue 29 Sep 2015 04:38:50 PM HKT
 *
 *         Author:  Huang Zonghao
 *          Email:  coding@huangzonghao.com
 *
 * =============================================================================
 */

#include "../include/cuda_support.h"

#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include "../thirdparty/nvidia/helper_cuda.h"
#include "../thirdparty/nvidia/helper_math.h"

#include "../include/cuda_support-inl.h"
#include "../include/demand_distribution.h"


/* =============================================================================
 *  The device kernels
 * =========================================================================== */



/* =============================================================================
 *  The global kernels
 * =========================================================================== */


/*
 * ===  GLOBAL KERNEL  =========================================================
 *         Name:  g_ZeroizeMemory
 *  Description:  zeroize the float array
 *       @param:  pointer to the array, length
 * =============================================================================
 */
__global__
void g_ZeroizeMemory(float *array, size_t length){
    size_t step_size = gridDim.x * blockDim.x;
    size_t myStartIdx = blockDim.x * blockIdx.x + threadIdx.x;
    for (size_t i = myStartIdx; i < length; i += step_size)
        array[i] = 0;

    __syncthreads();
    return;
}       /* -----  end of global kernel g_ZeroizeMemoryFloat  ----- */


__global__
void g_ZeroizeMemory(int *array, size_t length){
    size_t step_size = gridDim.x * blockDim.x;
    size_t myStartIdx = blockDim.x * blockIdx.x + threadIdx.x;
    for (size_t i = myStartIdx; i < length; i += step_size)
        array[i] = 0;

    __syncthreads();
    return;
}       /* -----  end of global kernel g_ZeroizeMemoryFloat  ----- */

/* =============================================================================
 *  The host functions
 * =========================================================================== */
/*
 * ===  FUNCTION  ==============================================================
 *         Name:  cuda_PassToDevice
 *  Description:  pass the array to device
 *       @param:  pointer to host array, pointer to device array, array size
 *      @return:  void
 * =============================================================================
 */
/* we cannot use const for any of the following address since that's not supported
 * but the cuda lib
 */
void cuda_PassToDevice ( float *h_array,
                         float *d_array,
                         size_t length ){

    checkCudaErrors(cudaMemcpy(d_array, h_array,
                               length * sizeof(float),
                               cudaMemcpyHostToDevice));
    return;
}       /* -----  end of function cuda_PassToDevice  ----- */

/* reload */
void cuda_PassToDevice ( float **h_array,
                         float **d_array,
                         size_t length ){

    checkCudaErrors(cudaMemcpy(d_array, h_array,
                               length * sizeof(float *),
                               cudaMemcpyHostToDevice));
    return;
}       /* -----  end of function cuda_PassToDevice  ----- */

/* reload */
void cuda_PassToDevice ( size_t *h_array,
                         size_t *d_array,
                         size_t length ){

    checkCudaErrors(cudaMemcpy(d_array, h_array,
                               length * sizeof(size_t),
                               cudaMemcpyHostToDevice));
    return;
}       /* -----  end of function cuda_PassToDevice  ----- */

/* reload */
void cuda_PassToDevice ( struct DemandDistribution *h_array,
                         struct DemandDistribution *d_array,
                         size_t length ){

    checkCudaErrors(cudaMemcpy(d_array, h_array,
                               length * sizeof(struct DemandDistribution),
                               cudaMemcpyHostToDevice));
    return;
}       /* -----  end of function cuda_PassToDevice  ----- */

void cuda_PassToDevice ( struct DemandDistribution **h_array,
                         struct DemandDistribution **d_array,
                         size_t length ){

    checkCudaErrors(cudaMemcpy(d_array, h_array,
                               length * sizeof(struct DemandDistribution*),
                               cudaMemcpyHostToDevice));
    return;
}       /* -----  end of function cuda_PassToDevice  ----- */

/*
 * ===  FUNCTION  ==============================================================
 *         Name:  cuda_ReadFromDevice
 *  Description:  read the array from device
 *       @param:  see pass_to_array
 *      @return:  see pass_to_array
 * =============================================================================
 */
void cuda_ReadFromDevice ( float *h_array,
                           float *d_array,
                           size_t length ){

    checkCudaErrors(cudaMemcpy(h_array, d_array,
                               length * sizeof(float),
                               cudaMemcpyDeviceToHost));

    return ;
}       /* -----  end of function cuda_ReadFromDevice  ----- */

/* reload */
void cuda_ReadFromDevice ( size_t *h_array,
                           size_t * d_array,
                           size_t length ){

    checkCudaErrors(cudaMemcpy(h_array, d_array,
                               length * sizeof(size_t),
                               cudaMemcpyDeviceToHost));

    return ;
}       /* -----  end of function cuda_ReadFromDevice  ----- */

void cuda_ReadFromDevice ( int *h_array,
                           int * d_array,
                           size_t length ){

    checkCudaErrors(cudaMemcpy(h_array, d_array,
                               length * sizeof(int),
                               cudaMemcpyDeviceToHost));

    return ;
}       /* -----  end of function cuda_ReadFromDevice  ----- */

/*
 * ===  FUNCTION  ==============================================================
 *         Name:  cuda_AllocateMemory
 *  Description:  allocate the memory to the given pointer (the data type is
 *                  float since we only need to take care of the value table
 *                  and the distribution table)
 *       @param:  length
 *      @return:  float*
 * =============================================================================
 */
float *cuda_AllocateMemoryFloat(size_t length){
    float *temp;
    checkCudaErrors(cudaMalloc(&temp, length * sizeof(float)));
    return temp;
}       /* -----  end of function cuda_AllocateMemoryFLoat  ----- */


int *cuda_AllocateMemoryInt(size_t length){
    int *temp;
    checkCudaErrors(cudaMalloc(&temp, length * sizeof(int)));
    return temp;
}

float **cuda_AllocateMemoryFloatPtr(size_t length){
    float **temp;
    checkCudaErrors(cudaMalloc(&temp, length * sizeof(float*)));
    return temp;
}

struct DemandDistribution *cuda_AllocateMemoryDemandDistribution(size_t length){
    struct DemandDistribution *temp;
    checkCudaErrors(cudaMalloc(&temp, length * sizeof(struct DemandDistribution)));
    return temp;
}

struct DemandDistribution **cuda_AllocateMemoryDemandDistributionPtr(size_t length){
    struct DemandDistribution **temp;
    checkCudaErrors(cudaMalloc(&temp, length * sizeof(struct DemandDistribution*)));
    return temp;
}

/*
 * ===  FUNCTION  ==============================================================
 *         Name:  cuda_ZeroizeMemoryFloat
 *  Description:  zeroize the float array
 *       @param:  pointer to array, array length
 *      @return:  success or not
 * =============================================================================
 */
/*
 * bool cuda_ZeroizeMemoryFloat(float *array, size_t length){
 *     <+body+>
 *         return <+return value+>;
 * }       [> -----  end of function cuda_ZeroizeMemoryFloat  ----- <]
 */

/*
 * ===  FUNCTION  ==============================================================
 *         Name:  cuda_FreeMemory
 *  Description:  free the cuda memory holding by the given pointer
 *       @param:  pointer
 *      @return:  none
 * =============================================================================
 */
void cuda_FreeMemory(float *ptr){
    checkCudaErrors(cudaFree(ptr));
    return;
}

void cuda_FreeMemory(float **ptr){
    checkCudaErrors(cudaFree(ptr));
    return;
}

void cuda_FreeMemory(struct DemandDistribution *ptr){
    checkCudaErrors(cudaFree(ptr));
    return;
}

void cuda_FreeMemory(struct DemandDistribution **ptr){
    checkCudaErrors(cudaFree(ptr));
    return;
}
/* -----  end of function cuda_FreeMemory  ----- */

/*
 * ===  FUNCTION  ==============================================================
 *         Name:  cuda_CheckGPU
 *  Description:  returns the number of devices, number of blocks per device,
 *                  and number of threads per block
 *       @param:  the pointer to the three parameters
 *      @return:  none
 * =============================================================================
 */
bool cuda_CheckGPU(int *num_devices, int *num_cores, int *core_size){

    /* int *get_device; */
    /* cudaGetDevice(get_device); */
    /* printf("the current device is %d\n", *get_device); */

    cudaGetDeviceCount(num_devices);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    *num_cores = _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) *\
                                      deviceProp.multiProcessorCount;
    *core_size = deviceProp.maxThreadsPerBlock;
    return true;
}       /* -----  end of function cuda_CheckGPU  ----- */

/* =============================================================================
 *                         end of file cuda_support.cu
 * =============================================================================
 */
