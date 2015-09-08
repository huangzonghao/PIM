/*
 * =============================================================================
 *
 *       Filename:  model_support.cu
 *
 *    Description:  The cuda supporting functions related to the algorithm
 *
 *        Created:  Sat Aug  8 15:35:08 2015
 *       Modified:  Mon 07 Sep 2015 10:51:58 AM HKT
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

#include "../include/cuda_support.cuh"
#include "../include/model_support-inl.cuh"
#include "../include/device_parameters.h"
#include "../include/demand_distribution.h"
#include "../include/command_queue.h"
#include "../include/system_info.h"

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
/* :TODO:Sun Aug  9 09:44:27 2015:huangzonghao:
 *  increase amount
 *  when the deplete_amount is smaller than 0
 */
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
 *  Description:  update one state value given z, q, and the DemandDistribution
 *       @param:  today's mD_index, today's storage, z, q, and parameters
 *      @return:  the expected value of today's storage under certain demand
 * =============================================================================
 */
__device__
float d_StateValue(float *table_for_reference,
                   int *mD_index,
                   size_t storage_today,
                   int z,
                   int q,
                   int demand_distri_idx,
                   struct DeviceParameters *d
                   ){

    float profit = 0;
    float sum    = 0;
    int *mD_temp = new int[d->m];
    DemandDistribution demand = *(d->demand_distributions[demand_distri_idx]);
    for ( int i = demand.min_demand; i < demand.max_demand; ++i){
        for (int j = 0; j < d->m; ++j){
            mD_temp[j] = mD_index[j];
        }
        profit = d->s * z /* depletion income */
               - d->h * max(int(storage_today) - z , 0) /* holding cost */
               - d->alpha * d->c * q /* ordering cost */
               + d->alpha * d->r * min(i, int(storage_today) - z + q) /* sale income */
               - d->alpha * d->theta * max(mD_index[0] - z - i, 0) /* disposal cost */
               + d->alpha * table_for_reference[d_GetTomorrowIndex(mD_temp, z+i-q, d->m)]; /* value of tomorrow */

        sum += profit * demand.table[i];
    }
    delete mD_temp;
    return sum;
}       /* -----  end of device kernel d_StateValue  ----- */
/*
 * ===  DEVICE KERNEL  =========================================================
 *         Name:  d_StateValueUpdate
 *  Description:  calculate the maximum expected state value for a given state
 *                  with the range of z and q, and also update the value in the
 *                  global value table
 *       @param:  the table to update, the table for reference, the global index
 *                   (dataIdx) of the current state, the range of z and q, and
 *                   the index of the distribution and the DeviceParameters
 *      @return:  none
 * =============================================================================
 */
__device__
void d_StateValueUpdate( float *table_to_update,
                         float *table_for_reference,
                         size_t dataIdx,
                         int *z_records,
                         int *q_records,
                         int min_z,
                         int max_z,
                         int min_q,
                         int max_q,
                         int demand_distri_idx,
                         struct DeviceParameters *d ){

    // Allocate a memory buffer on stack
    // So we don't need to do it for every loop
    // Last dimension are used to store the ordering
    // which could be used for sale
    int *mDarray = new int[d->m];

    /* temp_z and temp_q are to store the best z and q, for future use */
    int   temp_z         = 0;
    int   temp_q         = 0;
    float max_value      = 0.0;
    float expected_value = 0.0;
    size_t storage_today = d_decode(dataIdx, d->m, d->k, mDarray );

    for (int i_z = min_z; i_z <= max_z; ++i_z) {
        for (int i_q = min_q; i_q <= max_q; ++i_q) {
            expected_value = d_StateValue(table_for_reference,
                                          mDarray,
                                          storage_today,
                                          i_z,
                                          i_q,
                                          demand_distri_idx,
                                          d);

            // Simply taking the moving maximum
            if (expected_value > max_value + 1e-6) {
                max_value = expected_value;
                temp_z = i_z;
                temp_q = i_q;
            }
        }
    }

    // Store the optimal point and value
    table_to_update[dataIdx] = max_value;

    if(z_records != NULL)
        z_records[dataIdx] = temp_z;
    if(q_records != NULL)
        q_records[dataIdx] = temp_q;

    delete mDarray;
}    /* -----  end of device kernel d_StateValueUpdate  ----- */

/* =============================================================================
 *  The global kernels
 * =========================================================================== */

/*
 * ===  GLOBAL KERNEL  =========================================================
 *         Name:  g_ModelInit
 *  Description:  the kernel to init the value table.
 *       @param:  the DeviceParameters and the pointer to the value table
 * =============================================================================
 */
__global__
void g_ModelInit(struct DeviceParameters d, float *value_table){
    // the total number of threads which have been assigned for this task,
    // oneD layout everywhere
    size_t step_size = gridDim.x * blockDim.x;
    size_t myStartIdx = blockIdx.x * blockDim.x + threadIdx.x;

    for (size_t i = myStartIdx; i < d.table_length; i += step_size){
        value_table[i] = d_check_storage(i, d.m, d.k) * d.s;
    }
    __syncthreads();
    return;
}       /* -----  end of global kernel g_ModelInit  ----- */
/* =============================================================================
 *  The host functions
 * =========================================================================== */

/*
 * ===  FUNCTION  ==============================================================
 *         Name:  ModelInit
 *  Description:  to initialize the value table at the beginning of the program
 *                  for all policies, we just sell out all the items with the
 *                  salvage price
 *       @param:  the CommandQueue, the SystemInfo, and the pointer to the value
 *                   table
 *      @return:  success or not
 * =============================================================================
 */
bool ModelInit(CommandQueue *cmd, SystemInfo *sysinfo, float *value_table){
    g_ModelInit<<<sysinfo->get_value("num_cores"), sysinfo->get_value("core_size")>>>
                        (cmd->get_device_param_struct(), value_table);
    return true;
}       /* -----  end of function ModelInit  ----- */

/*
 * ===  FUNCTION  ==============================================================
 *         Name:  DeclareValueTable
 *  Description:  declare the two value tables in the cuda device
 *       @param:  table length
 *      @return:  float**
 * =============================================================================
 */
float **DeclareValueTable(size_t table_length, SystemInfo* sysinfo){
    /* first declare the host space for the two pointers holding the two tables */
    float **temp = new float*[2];
    /* then the device memory space */
    for (int i = 0; i < 2; ++i){
        checkCudaErrors(cudaMalloc(temp + i, table_length * sizeof(float)));
        /* then zeroize the table */
        g_ZeroizeMemoryFloat
            <<<sysinfo->get_value("num_cores"), sysinfo->get_value("core_size")>>>
            (temp[i], table_length);
    }
    return temp;
}       /* -----  end of function DeclareValueTable  ----- */

/*
 * ===  FUNCTION  ==============================================================
 *         Name:  CleanUpValueTable
 *  Description:  clean up the memory space declared in DeclareValueTable
 *                  both host memory and the device memory
 *       @param:  float** and table length
 *      @return:  success or not
 * =============================================================================
 */
bool CleanUpValueTable(float **value_tables, size_t table_length){
    /* first the device memory space */
    for (int i = 0; i < 2; ++i){
        checkCudaErrors(cudaFree(value_tables[i]));
    }
    /* then the host memory */
    delete value_tables;
    value_tables = NULL;
    return true;
}       /* -----  end of function CleanUpValueTable  ----- */


/* =============================================================================
 *                         end of file model_support.cu
 * =============================================================================
 */
