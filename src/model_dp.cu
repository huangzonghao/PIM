/*
 * =============================================================================
 *
 *       Filename:  model_dp.cu
 *
 *    Description:  All the functions to compute with the dynamic programming
 *                    algorithm
 *
 *        Created:  Fri Aug  7 23:47:24 2015
 *       Modified:  Fri Aug 28 10:58:14 2015
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
void g_ModelDP(float *table_to_update,
               float *table_for_reference,
               int demand_distri_idx,
               int *z_records,
               int *q_records,
               size_t level_size,
               size_t batch_idx,
               DeviceParameters d){

    size_t myIdx = threadIdx.x + blockIdx.x * blockDim.x;

    if (myIdx < level_size) {

        size_t dataIdx = batch_idx * level_size + myIdx;
        size_t parentIdx = dataIdx - level_size;

        if (dataIdx == 0 || z_records[parentIdx] == 0) {
            d_StateValueUpdate(table_to_update,
                               table_for_reference,
                               dataIdx,
                               z_records, q_records,
                               /* [min_z, max_z] */
                               0, 2,
                               /* [min_q, max_q] */
                               0, d.k - 1,
                               demand_distri_idx, d);
        }
        else /* (depletion[parent] != 0) */ {
            d_StateValueUpdate(table_to_update,
                               table_for_reference,
                               dataIdx,
                               z_records, q_records,
                               /* [min_z, max_z] */
                               z_records[parentIdx] + 1,
                               z_records[parentIdx] + 2,
                               /* [min_q, max_q] */
                               q_records[parentIdx],
                               q_records[parentIdx] + 1,
                               demand_distri_idx, d);
        }
    }
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
bool ModelDP(CommandQueue *cmd,
             SystemInfo *sysinfo,
             float *table_to_update,
             float *table_for_reference,
             int demand_distri_idx,
             int *z, int *q){

    size_t level_size = pow(cmd->get_h_param("k"), cmd->get_h_param("m"));
    // The very first state 0,0,...,0
    g_ModelDP<<<1, 1>>>(  table_to_update,
                          table_for_reference,
                          demand_distri_idx,
                          z, q,
                          1, 0,
                          *(cmd->get_device_param_pointer) );

    size_t num_blocks_used;
    size_t core_size = sysinfo->get_value("core_size");
    for (size_t i_level = 0; i_level < cmd->get_h_param("m"); ++i_level) {
        num_blocks_used = i_level * cmd->get_h_param("k");
        level_size = pow(cmd->get_h_param("k"), i_level);
        for (size_t i_batch = 1; i_batch < cmd->get_h_param("k"); i_batch++) {
            g_ModelDP<<<num_blocks_used, core_size >>>(  table_to_update,
                                                         table_for_reference,
                                                         demand_distri_idx,
                                                         z, q,
                                                         level_size,
                                                         i_batch,
                                                         *(cmd->get_device_param_pointer) );
        }
    }
    return true;
}       /* -----  end of function ModelDP  ----- */


/* =============================================================================
 *                         end of file model_dp.cu
 * =============================================================================
 */
