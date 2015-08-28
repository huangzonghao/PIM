/*
 * =============================================================================
 *
 *       Filename:  model_fluid.cu
 *
 *    Description:  All the functions to compute the fluid policy
 *
 *        Created:  Fri Aug  7 23:34:03 2015
 *       Modified:  Fri Aug 28 09:25:00 2015
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
 *         Name:  g_ModelFluid
 *  Description:  the kernel to update the value table for one day with fluid
 *                  fluid policy.
 *       @param:  the DeviceParameters, pointer to the current table, pointer
 *                   to the last table
 * =============================================================================
 */
__global__
void g_ModelFluid(struct DeviceParameters d,
                  float *table_to_update,
                  float *table_for_reference,
                  int demand_distri_idx,
                  size_t depletion_indicator ){

    // this is both the thread index and the data index in this batch
    size_t myIdx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t dataIdx = myIdx + batchIdx * gridDim.x * blockDim.x;

    storage_today = d_decode(dataIdx, d.m, d.k, mD_index );

    /* because we may use more threads than needed */
    if(dataIdx < d.table_length){
        if(depletion_indicator){ // the last day
            d_StateValueUpdate( table_to_update,
                                table_for_reference,
                                dataIdx,
                                NULL, NULL,
                                /* [min_z, max_z] */
                                depletion_indicator * d.T, depletion_indicator * d.T,
                                /* [min_q, max_q] */
                                0, d.k - 1,
                                demand_distri_idx, /* the index of the demand distribution */
                                d);
        }
        else{
            d_StateValueUpdate( table_to_update,
                                table_for_reference,
                                dataIdx,
                                NULL, NULL,
                                /* [min_z, max_z] */
                                0, 0,
                                /* [min_q, max_q] */
                                0, d.k - 1,
                                demand_distri_idx, /* the index of the demand distribution */
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
                float *table_to_update,
                float *table_for_reference,
                int demand_distri_idx,
                size_t depletion_indicator){

    // each thread will take care of a state at once
    size_t batch_amount = d.table_length / sysinfo->get_value("num_cores") /
                        sysinfo->get_value("core_size") + 1;

    for ( size_t i = 0; i < batch_amount; ++i){
        g_ModelFluid
            <<<sysinfo->get_value("num_cores"), sysinfo->get_value("core_size")>>>
            (d,
             table_to_update,
             table_for_reference,
             demand_distri_idx,
             depletion_indicator);
    }
    return true;
}       /* -----  end of function ModelFluid  ----- */



/* =============================================================================
 *                         end of file model_fluid.cu
 * =============================================================================
 */
