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
 *       Modified:  Sun Aug  9 10:18:39 2015
 *
 *         Author:  Huang Zonghao
 *          Email:  coding@huangzonghao.com
 *
 * =============================================================================
 */

#include "../include/cuda_support.h"

#include <stdlib.h>
#include <cuda.h>
#include "../thirdparty/nvidia/helper_cuda.h"
#include "../thirdparty/nvidia/helper_math.h"

#include "../include/cuda_support-inl.h"

/* :TODO:Wed Jul 29 11:25:24 2015:huangzonghao:
 *  a function to report the system configuration,
 *
 */

/*
 * ===  FUNCTION  ==============================================================
 *         Name:  cuda_PassToDevice
 *  Description:  pass the array to device
 *       @param:  pointer to host array, pointer to device array, array size
 *      @return:  void
 * =============================================================================
 */
void cuda_PassToDevice ( const float* h_array, const float* d_array,\
                             size_t length ){
    checkCudaErrors(cudaMemcpy(d_array, h_array,\
                               length * sizeof(float),\
                               cudaMemcpyHostToDevice));
    return;
}       /* -----  end of function cuda_PassToDevice  ----- */

void cuda_PassToDevice ( const float ** h_array, const float ** d_array,\
                             size_t length ){
    checkCudaErrors(cudaMemcpy(d_array, h_array,\
                               length * sizeof(float *),\
                               cudaMemcpyHostToDevice));
    return;
}       /* -----  end of function cuda_PassToDevice  ----- */

/* reload */
void cuda_PassToDevice ( const size_t * h_array, const size_t * d_array,\
                             size_t length ){
    checkCudaErrors(cudaMemcpy(d_array, h_array,\
                               length * sizeof(size_t),\
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
void cuda_ReadFromDevice ( const float* h_array, const float* d_array,\
                               size_t length ){
    checkCudaErrors(cudaMemcpy(h_array, d_array,\
                               length * sizeof(float),\
                               cudaMemcpyDeviceToHost));

    return ;
}       /* -----  end of function cuda_ReadFromDevice  ----- */
/* reload */
void cuda_ReadFromDevice ( const size_t * h_array, const size_t * d_array,\
                               size_t length ){
    checkCudaErrors(cudaMemcpy(h_array, d_array,\
                               length * sizeof(size_t),\
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
float * cuda_AllocateMemoryFLoat(int length){
    float * temp;
    checkCudaErrors(cudaMalloc(&temp, length * sizeof(float)));
    return temp;
}       /* -----  end of function cuda_AllocateMemoryFLoat  ----- */

float ** cuda_AllocateMemoryFloatPtr(int length){
    float ** temp;
    checkCudaErrors(cudaMalloc(&temp, length * sizeof(float*)));
    return temp;
}
/*
 * ===  FUNCTION  ==============================================================
 *         Name:  cuda_FreeMemory
 *  Description:  free the cuda memory holding by the given pointer
 *       @param:  pointer
 *      @return:  none
 * =============================================================================
 */
void cuda_FreeMemory(float * ptr){
    checkCudaErrors(cudaFree(ptr));
    return;
}       /* -----  end of function cuda_FreeMemory  ----- */

/* =============================================================================
 *                         end of file cuda_support.cu
 * =============================================================================
 */
