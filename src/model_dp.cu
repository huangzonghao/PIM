/*
 * =============================================================================
 *
 *       Filename:  model_dp.cu
 *
 *    Description:  All the functions to compute with the dynamic programming
 *                    algorithm
 *
 *        Created:  Fri Aug  7 23:47:24 2015
 *       Modified:  Tue 29 Sep 2015 05:45:22 PM HKT
 *
 *         Author:  Huang Zonghao
 *          Email:  coding@huangzonghao.com
 *
 * =============================================================================
 */
#include "../include/models.h"
#include "../include/model_support.h"
#include "../include/model_support.cuh"
#include "../include/demand_distribution.h"
#include "../include/device_parameters.h"
#include "../include/command_queue.h"
#include "../include/system_info.h"

/* for debugging */
#include "../include/cuda_support.h"

/*
 * ===  GLOBAL KERNEL  =========================================================
 *         Name:  g_ModelDP
 *  Description:  the kernel function for the tree structure
 *       @param:  table to update, table for reference, amount to deplete, amount
 *                   to order, the total number of states of this level the
 *                   batch index(the value of the digit we are updating within
 *                   certain level)
 * =============================================================================
 */
__global__
void g_ModelDP( float *table_to_update,
                float *table_for_reference,
                int demand_distri_idx,
                int **md_spaces,
                int *z_records,
                int *q_records,
                size_t level_size,
                size_t batch_idx,
                struct DeviceParameters d){
    /* if(threadIdx.x < 1){ */
        /* if(state == 1){ */
            /* printf("i am happy \n"); */
        /* } */
        /* else {printf("i am sad\n");} */

        /* printf("reporting\n"); */
    /* } */
    size_t myIdx = threadIdx.x + blockIdx.x * blockDim.x;

    if (myIdx < level_size) {

        size_t dataIdx = batch_idx * level_size + myIdx;
        size_t parentIdx = dataIdx - level_size;

        if (dataIdx == 0 || z_records[parentIdx] == 0) {
            d_StateValueUpdate( table_to_update,
                                table_for_reference,
                                dataIdx,
                                md_spaces,
                                z_records, q_records,
                                /* [min_z, max_z] */
                                0, 1,
                                /* [min_q, max_q] */
                                0, d.k - 1,
                                demand_distri_idx, &d );
        }
        else{               /* (depletion[parent] != 0) */
            d_StateValueUpdate( table_to_update,
                                table_for_reference,
                                dataIdx,
                                md_spaces,
                                z_records, q_records,
                                /* [min_z, max_z] */
                                z_records[parentIdx] + 1,
                                z_records[parentIdx] + 1,
                                /* [min_q, max_q] */
                                q_records[parentIdx],
                                q_records[parentIdx],
                                demand_distri_idx, &d );
        }
    }
    __syncthreads();
    return;
}       /* -----  end of global kernel g_ModelDP  ----- */

/*
 * ===  FUNCTION  ==============================================================
 *         Name:  ModelDP
 *  Description:  to update the table for one period with the dynamic programming
 *                  algorithm
 *       @param:  control sequence, system information, the index of the period
 *      @return:  success or not
 * =============================================================================
 */
__global__
void tryhaha(int level, int batch){
    /* printf("i am %d, and there are %d cores, each has %d threads\n", threadIdx.x + blockIdx.x * blockDim.x, gridDim.x, blockDim.x); */
    if(threadIdx.x < 1 || threadIdx.x == 1023)
    /* printf("i am %d, this is %d level, %d batch\n", threadIdx.x + blockIdx.x * blockDim.x, level, batch); */
    printf("i am %d of %d, this is %d level, %d batch\n", threadIdx.x, blockIdx.x, level, batch);
    __syncthreads();
    return;
}
bool ModelDP(CommandQueue *cmd,
             SystemInfo *sysinfo,
             float *table_to_update,
             float *table_for_reference,
             int demand_distri_idx,
             int period_idx,
             int **md_spaces,
             int *z, int *q){


    // The very first state 0,0,...,0
    g_ModelDP<<<1, 1>>>( table_to_update,
                         table_for_reference,
                         demand_distri_idx,
                         md_spaces,
                         z, q,
                         1, 0,
                         cmd->get_device_param_struct());

    printf("mass states: \n");
    size_t level_size;
    /* size_t num_blocks_used; */
    size_t grid_size = sysinfo->get_value("num_cores");
    size_t core_size = sysinfo->get_value("core_size");
    printf("before calling the kernel\n");
    for (size_t i_level = 0; i_level < (size_t)cmd->get_h_param("m"); ++i_level) {
        /* num_blocks_used = i_level * (size_t)cmd->get_h_param("k"); */
        /* level_size is the number of threads to run at the same time */
        level_size = pow(cmd->get_h_param("k"), i_level);
        for (size_t i_batch = 1; i_batch < (size_t)cmd->get_h_param("k"); i_batch++) {
            /* printf("this is level %d, batch %d\n", i_level, i_batch); */
            /* tryhaha<<<num_blocks_used, core_size >>>(i_level, i_batch); */
            /* tryhaha<<<1, 1024 >>>(i_level, i_batch); */
            /* g_ModelDP<<<num_blocks_used, core_size>>>( table_to_update, */
            g_ModelDP<<<grid_size, core_size>>>( table_to_update,
                                                       table_for_reference,
                                                       demand_distri_idx,
                                                       md_spaces,
                                                       z, q,
                                                       level_size,
                                                       i_batch,
                                                       /* 0, */
                                                       cmd->get_device_param_struct());
        }
    }
    return true;
}       /* -----  end of function ModelDP  ----- */


/* =============================================================================
 *                         end of file model_dp.cu
 * =============================================================================
 */
