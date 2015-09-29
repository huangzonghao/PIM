/*
 * =============================================================================
 *
 *       Filename:  model_fluid.cu
 *
 *    Description:  All the functions to compute the fluid policy
 *
 *        Created:  Fri Aug  7 23:34:03 2015
 *       Modified:  Tue 29 Sep 2015 05:48:11 PM HKT
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
#include "../include/model_support-inl.cuh"
#include "../include/device_parameters.h"
#include "../include/command_queue.h"
#include "../include/demand_distribution.h"
#include "../include/system_info.h"


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
                  size_t depletion_indicator,
                  size_t batch_idx,
                  int **md_spaces ){

    // this is both the thread index and the data index in this batch
    size_t myIdx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t dataIdx = myIdx + batch_idx * gridDim.x * blockDim.x;

    /* because we may use more threads than needed */
    if(dataIdx < d.table_length){
        if(depletion_indicator){ // the last day
            d_StateValueUpdate( table_to_update,
                                table_for_reference,
                                dataIdx,
                                md_spaces,
                                NULL, NULL,
                                /* [min_z, max_z] */
                                depletion_indicator * d.T, depletion_indicator * d.T,
                                /* [min_q, max_q] */
                                0, d.k - 1,
                                demand_distri_idx, /* the index of the demand distribution */
                                &d);
        }
        else{
            d_StateValueUpdate( table_to_update,
                                table_for_reference,
                                dataIdx,
                                md_spaces,
                                NULL, NULL,
                                /* [min_z, max_z] */
                                0, 0,
                                /* [min_q, max_q] */
                                0, d.k - 1,
                                demand_distri_idx, /* the index of the demand distribution */
                                &d);
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
bool ModelFluid( CommandQueue *cmd,
                 SystemInfo *sysinfo,
                 float *table_to_update,
                 float *table_for_reference,
                 int demand_distri_idx,
                 /* size_t depletion_indicator, */
                 int period_idx,
                 int **md_spaces,
                 int *z_records, int *q_records ){

    /* caculate the deplletion_indicator in side */
    size_t depletion_indicator = 0;
    if(period_idx == 1){
        DemandDistribution *demand = cmd->get_h_demand_pointer(demand_distri_idx);
        float expect_demand = 0;
        for(int i = 0; i < (int)(demand->max_demand - demand->min_demand); ++i){
            expect_demand += (i + demand->min_demand) * demand->table[i];
        }
        depletion_indicator = (int)ceil(expect_demand);
    }

    // each thread will take care of a state at once
    size_t batch_amount = (size_t)cmd->get_d_param("table_length")
                        / sysinfo->get_value("num_cores")
                        / sysinfo->get_value("core_size") + 1;
    for ( size_t i = 0; i < batch_amount; ++i){
        g_ModelFluid
            <<<sysinfo->get_value("num_cores"), sysinfo->get_value("core_size")>>>
            /* something went wrong here */
            /* <<<sysinfo->get_value("num_cores"), 100>>> */
            (*(cmd->get_device_param_pointer()),
             table_to_update,
             table_for_reference,
             demand_distri_idx,
             depletion_indicator,
             i,
             md_spaces);
    }

    return true;
}       /* -----  end of function ModelFluid  ----- */



/* =============================================================================
 *                         end of file model_fluid.cu
 * =============================================================================
 */
