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
 *       Modified:  Sat Sep  5 10:55:48 2015
 *
 *         Author:  Huang Zonghao
 *          Email:  coding@huangzonghao.com
 *
 * =============================================================================
 */
#ifndef CUDA_SUPPORT_H_
#define CUDA_SUPPORT_H_

#include <stdlib.h>

struct DemandDistribution;

/* #####   EXPORTED FUNCTION DECLARATIONS   ################################## */
/*-----------------------------------------------------------------------------
 *  pass to device
 *-----------------------------------------------------------------------------*/
void cuda_PassToDevice ( const float *h_array,
                         const float *d_array,
                         size_t length );

void cuda_PassToDevice ( const float **h_array,
                         const float **d_array,
                         size_t length );

void cuda_PassToDevice ( const size_t *h_array,
                         const size_t *d_array,
                         size_t length );

void cuda_PassToDevice ( const DemandDistribution *h_array,
                         const DemandDistribution *d_array,
                         size_t length );

void cuda_PassToDevice ( DemandDistribution **h_array,
                         DemandDistribution **d_array,
                         size_t length );

/*-----------------------------------------------------------------------------
 *  read from device
 *-----------------------------------------------------------------------------*/
void cuda_ReadFromDevice ( const float *h_array,
                           const float *d_array,
                           size_t length );

void cuda_ReadFromDevice ( const size_t *h_array,
                           const size_t *d_array,
                           size_t length );

/*-----------------------------------------------------------------------------
 *  allocate memory
 *-----------------------------------------------------------------------------*/
float *cuda_AllocateMemoryFloat(int length);
int *cuda_AllocateMemoryInt(int length);
float **cuda_AllocateMemoryFloatPtr(int length);
DemandDistribution *cuda_AllocateMemoryDemandDistribution(size_t length);
DemandDistribution **cuda_AllocateMemoryDemandDistributionPtr(size_t length);

/*-----------------------------------------------------------------------------
 *  free memory
 *-----------------------------------------------------------------------------*/
void cuda_FreeMemory(float *ptr);
void cuda_FreeMemory(float **ptr);
void cuda_FreeMemory(DemandDistribution *ptr);
void cuda_FreeMemory(DemandDistribution **ptr);

/*-----------------------------------------------------------------------------
 *  check hardware
 *-----------------------------------------------------------------------------*/
bool cuda_CheckGPU(int*, int*, int*);






#endif   /* ----- #ifndef CUDA_SUPPORT_H_  ----- */

/* =============================================================================
 *                         end of file cuda_support.h
 * =============================================================================
 */
