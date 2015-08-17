/*
 * =============================================================================
 *
 *       Filename:  cuda_support.h
 *
 *    Description:  The header file of cuda cuda_support.cu
 *                    And this flie will be included by .cc files only
 *                    The declaration of shared kernels can be found in
 *                    cuda_support-ker.h
 *
 *        Created:  Thu Jul 23 03:40:09 2015
 *       Modified:  Thu Aug 13 11:53:14 2015
 *
 *         Author:  Huang Zonghao
 *          Email:  coding@huangzonghao.com
 *
 * =============================================================================
 */
#ifndef CUDA_SUPPORT_H_
#define CUDA_SUPPORT_H_

#include <stdlib.h>

/* #####   EXPORTED FUNCTION DECLARATIONS   ################################## */
void cuda_PassToDevice ( const float *h_array, const float *d_array,\
                             size_t length );
void cuda_PassToDevice ( const size_t *h_array, const size_t *d_array,\
                             size_t length );
void cuda_ReadFromDevice ( const float *h_array, const float *d_array,\
                               size_t length );
void cuda_ReadFromDevice ( const size_t *h_array, const size_t *d_array,\
                               size_t length );
float* cuda_AllocateMemoryFloat(int length);
float** cuda_AllocateMemoryFloatPtr(int length);
void cuda_FreeMemory(float *ptr);
void cuda_FreeMemory(float **ptr);
bool cuda_CheckGPU(int*, int*, int*);






#endif   /* ----- #ifndef CUDA_SUPPORT_H_  ----- */

/* =============================================================================
 *                         end of file cuda_support.h
 * =============================================================================
 */
