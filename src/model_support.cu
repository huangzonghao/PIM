/*
 * =============================================================================
 *
 *       Filename:  model_support.cu
 *
 *    Description:  The cuda supporting functions related to the algorithm
 *
 *        Created:  Sat Aug  8 15:35:08 2015
 *       Modified:  Sun Aug  9 10:25:06 2015
 *
 *         Author:  Huang Zonghao
 *          Email:  coding@huangzonghao.com
 *
 * =============================================================================
 */

#include "../include/model_support.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include "../thirdparty/nvidia/helper_cuda.h"
#include "../thirdparty/nvidia/helper_math.h"

/* =============================================================================
 *  The device kernels
 * =========================================================================== */

/*
 * ===  DEVICE KERNEL  =========================================================
 *         Name:  d_DepleteStorage
 *  Description:  deplete the storage by the certain amount
 *       @param:  mD_index, amount to deplete, m
 *      @return:  none
 * =============================================================================
 */
__device__
void d_DepleteStorage(int * mD_index, size_t deplete_amount, size_t m){
    size_t buffer = 0;
    size_t i = 0;
    if (deplete_amount > 0) {
        while(!deplete_amount && i < m){
            if ( !mD_index[i]){
                ++i;
                continue;
            }
            if(mD_index[i] >= deplete_amount )
            {
                mD_index[i] -= deplete_amount;
                deplete_amount  = 0;
                break;
            }
            buffer = deplete_amount - mD_index[i];
            mD_index[i] = 0;
            deplete_amount = buffer;
            buffer = 0;
            ++i;
        }
    }
    else if(deplete_amount < 0){
/* :TODO:Sun Aug  9 09:44:27 2015:huangzonghao:
 *  increase amount
 */
    }
}       /* -----  end of device kernel d_DepleteStorage  ----- */

/*
 * ===  DEVICE KERNEL  =========================================================
 *         Name:  d_GetTomorrowIndex
 *  Description:  calculate the index of Tomorrow's state with today's state
 *                  and today's state change(get from z, q, number of items sold)
 *       @param:  today's state mDarray, today's state change(deplete as positive)
 *                   and m
 *      @return:  the data index of tomorrow's state
 * =============================================================================
 */
__device__
size_t d_GetTomorrowIndex(int * mD_index, int today_deplete, size_t m){
    d_DepleteStorage(mD_index, today_deplete, m);
    return d_check_storage(mD_index, m);
}       /* -----  end of device kernel d_GetTomorrowIndex  ----- */
/*
 * ===  DEVICE KERNEL  =========================================================
 *         Name:  d_StateValue
 *  Description:  calculate the state value for given z and q
 *       @param:  today's mD_index, today's storage, z, q, and parameters
 *      @return:  the expected value of today's storage under certain demand
 * =============================================================================
 */
__device__
float d_StateValue(float * last_table,
                   int * mD_index,
                   size_t storage_today,
                   int z,
                   int q,
                   struct DeviceParameters &d,
                   int demand_table_idx){
    float profit = 0;
    float sum    = 0;
    int * mD_temp = int[d.m];
    for ( int i = d.min_demand; i < d.max_demand; ++i){
        for (int i = 0; i < d.m; ++i){
            mD_temp[i] = mD_index[i];
        }
        profit = d.s * z\
               - d.h * max(int(int(storage_today) - z) , 0)\
               - d.alpha * d.c * q\
               + d.alpha * d.r * min(int(i), int(storage_today) - z + q)\
               - d.alpha * d.theta * max(mD_index[0] - z - i, 0)\
               + d.alpha * last_table[d_GetTomorrowIndex(mD_temp, z+i-q, d.m)];

        sum += profit * d.demand_distributions[demand_table_idx][i];
    }
    delete mD_temp;
    return sum;
}       /* -----  end of device kernel d_StateValue  ----- */
/* =============================================================================
 *  The global kernels
 * =========================================================================== */
/* =============================================================================
 *  The host functions
 * =========================================================================== */



/* =============================================================================
 *                         end of file model_support.cu
 * =============================================================================
 */
