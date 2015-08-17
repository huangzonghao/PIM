/*
 * =============================================================================
 *
 *       Filename:  model_support-inl.cuh
 *
 *    Description:  The inline cuda supporting functions related to the
 *                    algorithm
 *
 *        Created:  Sat Aug  8 15:28:06 2015
 *       Modified:  Sun Aug  9 16:22:24 2015
 *
 *         Author:  Huang Zonghao
 *          Email:  coding@huangzonghao.com
 *
 * =============================================================================
 */
#ifndef MODEL_SUPPORT_INL_CUH_
#define MODEL_SUPPORT_INL_CUH_
#include <cuda.h>
#include <cuda_runtime.h>

#include "../thirdparty/nvidia/helper_cuda.h"

__device__ inline
size_t d_decode (size_t oneDIdx, size_t m, size_t k, size_t* mDIdx){
    size_t sum  = 0;
    size_t temp = 0;
    for( size_t i = 0; i < m ; ++i){
        temp = oneDIdx % k;
        mDIdx[m - 1 - i] = temp;
        sum += temp;
        oneDIdx /= k;
    }
    return sum;
}

__device__ inline
size_t d_check_storage(size_t* mDarray, size_t m){
    size_t result = 0;
    for (size_t i = 0; i < m ; ++i ){
            result += mDarray[i];
    }
    return result;
}

__device__ inline
size_t d_check_storage(size_t oneDIdx, size_t m, size_t k){
    size_t result = 0;
    for (size_t i = 0; i < m ; ++i ){
        result += oneDIdx % k;
        oneDIdx /= k;
    }
    return result;
}
#endif   /* ----- #ifndef MODEL_SUPPORT-INL_H_  ----- */
/* =============================================================================
 *                         end of file model_support-inl.cuh
 * =============================================================================
 */
