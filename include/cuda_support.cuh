/*
 * =============================================================================
 *
 *       Filename:  cuda_support.cuh
 *
 *    Description:  The declarations of the kernels defined in cuda_support.cu
 *                    And this file can only be included in .cu files
 *
 *        Created:  Sun Aug  9 16:18:29 2015
 *       Modified:  Sun Aug  9 16:21:15 2015
 *
 *         Author:  Huang Zonghao
 *          Email:  coding@huangzonghao.com
 *
 * =============================================================================
 */
#ifndef CUDA_SUPPORT_CUH_
#define CUDA_SUPPORT_CUH_

__global__
void g_ZeroizeMemory(float *array, size_t length);
__global__
void g_ZeroizeMemory(int *array, size_t length);


#endif   /* ----- #ifndef CUDA_SUPPORT_CUH_  ----- */
/* =============================================================================
 *                         end of file cuda_support.cuh
 * =============================================================================
 */
