/*
 * =============================================================================
 *
 *       Filename:  model_fluid.cu
 *
 *    Description:  All the functions to compute the fluid policy
 *
 *        Created:  Fri Aug  7 23:34:03 2015
 *       Modified:  Wed Aug 26 16:28:54 2015
 *
 *         Author:  Huang Zonghao
 *          Email:  coding@huangzonghao.com
 *
 * =============================================================================
 */
#include "../include/models.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include "../thirdparty/nvidia/helper_cuda.h"

#include "../include/model_support.cuh"

/*
 * ===  GLOBAL KERNEL  =========================================================
 *         Name:  g_ModelFluidInit
 *  Description:  the kernel to init the value tables for model fluid
 *                  all the items have been sold in the last day
 *       @param:  the DeviceParameters, pointer to the value table
 * =============================================================================
 */
__global__
void g_ModelFluidInit(struct DeviceParameters d, float *value_table){
    // the total number of threads which have been assigned for this task,
    // oneD layout everywhere
    size_t step_size = gridDim.x * blockDim.x;
    size_t myStartIdx = blockIdx.x * blockDim.x + threadIdx.x;

    for (size_t i = myStartIdx; i < d.table_length; i += step_size){
        value_table[i] = d_check_storage(i, d.m, d.k) * d.s;
    }
    __syncthreads();
    return;
}       /* -----  end of global kernel g_ModelFluidInit  ----- */

/*
 * ===  FUNCTION  ==============================================================
 *         Name:  ModelFluidInit
 *  Description:  the initialization function ModelFluid
 *       @param:  the control sequence, the system information and the pointer
 *                   to the value table
 *      @return:  success or not
 * =============================================================================
 */
bool ModelFluidInit(CommandQueue *cmd, SystemInfo *sysinfo, float *value_table){
    g_ModelFluidInit<<<sysinfo->get_value["num_cores"],\
                        sysinfo->get_value["core_size"]>>>\
                        (*(cmd->get_device_param_pointer), value_table);
    return true;
}       /* -----  end of function ModelFluidInit  ----- */

/*
 * ===  GLOBAL KERNEL  =========================================================
 *         Name:  g_ModelFluid
 *  Description:  the kernel to update the value table for one day with fluid
 *                  fluid policy.
 *       @param:  the DeviceParameters, pointer to the current table, pointer
 *                   to the last table
 * =============================================================================
 */
__global__
void g_ModelFluid(struct DeviceParameters d,
                  float *current_table,
                  float *last_table,
                  size_t depletion_indicator ){

    // this is both the thread index and the data index in this batch
    size_t myIdx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t dataIdx = myIdx + batchIdx * gridDim.x * blockDim.x;

    storage_today = d_decode(dataIdx, d.m, d.k, mD_index );

    /* because we may use more threads than needed */
    if(dataIdx < d.table_length){
        if(depletion_indicator){ // the last day
            d_StateValueUpdate( current_table,
                                last_table,
                                dataIdx,
                                NULL, NULL,
                                /* [min_z, max_z] */
                                depletion_indicator * d.T, depletion_indicator * d.T,
                                /* [min_q, max_q] */
                                0, d.k - 1,
                                0, /* the index of the demand distribution */
                                d);
        }
        else{
            d_StateValueUpdate( current_table,
                                last_table,
                                dataIdx,
                                NULL, NULL,
                                /* [min_z, max_z] */
                                0, 0,
                                /* [min_q, max_q] */
                                0, d.k - 1,
                                0, /* the index of the demand distribution */
                                d);
        }
    }
    return;
}       /* -----  end of global kernel g_ModelFluid  ----- */

/*
 * ===  FUNCTION  ==============================================================
 *         Name:  ModelFluid
 *  Description:  to update the table for one day with policy fluid
 *       @param:  the control sequence, the system information and the index of
 *                   the day and the depletion_indicator(zero means no depletion
 *                   and positive integer means the expected demand)
 *      @return:  success or not
 * =============================================================================
 */
bool ModelFluid(CommandQueue *cmd,
                SystemInfo *sysinfo,
                float *value_table,
                int current_table_idx,
                size_t depletion_indicator){

    // each thread will take care of a state at once
    size_t batch_amount = d.table_length / sysinfo->get_value["num_cores"] /\
                        sysinfo->get_value["core_size"] + 1;

    for ( size_t i = 0; i < batch_amount; ++i){
        g_ModelFluid
            <<<sysinfo->get_value["num_cores"], sysinfo->get_value["core_size"]>>>\
            (d,
             value_table[current_table_idx],
             value_table[1 - current_table_idx],
             depletion_indicator);
    }
    return true;
}       /* -----  end of function ModelFluid  ----- */



/* =============================================================================
 *                         end of file model_fluid.cu
 * =============================================================================
 */
