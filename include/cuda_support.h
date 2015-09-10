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
 *       Modified:  Thu 10 Sep 2015 11:45:02 AM HKT
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
void cuda_PassToDevice ( float *h_array,
                         float *d_array,
                         size_t length );

void cuda_PassToDevice ( float **h_array,
                         float **d_array,
                         size_t length );

void cuda_PassToDevice ( size_t *h_array,
                         size_t *d_array,
                         size_t length );

void cuda_PassToDevice ( struct DemandDistribution *h_array,
                         struct DemandDistribution *d_array,
                         size_t length );

void cuda_PassToDevice ( struct DemandDistribution **h_array,
                         struct DemandDistribution **d_array,
                         size_t length );

/*-----------------------------------------------------------------------------
 *  read from device
 *-----------------------------------------------------------------------------*/
void cuda_ReadFromDevice ( float *h_array,
                           float *d_array,
                           size_t length );

void cuda_ReadFromDevice ( size_t *h_array,
                           size_t *d_array,
                           size_t length );

/*-----------------------------------------------------------------------------
 *  allocate memory
 *-----------------------------------------------------------------------------*/
float *cuda_AllocateMemoryFloat(size_t length);
int *cuda_AllocateMemoryInt(size_t length);
float **cuda_AllocateMemoryFloatPtr(size_t length);
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
