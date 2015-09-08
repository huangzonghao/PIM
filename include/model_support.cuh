/*
 * =============================================================================
 *
 *       Filename:  model_support.cuh
 *
 *    Description:  This header file contains the cuda related declarations of
 *                    model_support.cu and shall be included by another .cu file
 *
 *        Created:  Mon Aug 10 18:36:13 2015
 *       Modified:  Mon Aug 10 18:36:13 2015
 *
 *         Author:  Huang Zonghao
 *          Email:  coding@huangzonghao.com
 *
 * =============================================================================
 */
#ifndef MODEL_SUPPORT_CUH_
#define MODEL_SUPPORT_CUH_
__device__
void d_DepleteStorage(int *mD_index, size_t deplete_amount, size_t m);

__device__
size_t d_GetTomorrowIndex(int *mD_index, int today_deplete, size_t m);

__device__
float d_StateValue(float *last_table,
                   int *mD_index,
                   size_t storage_today,
                   int z,
                   int q,
                   int demand_table_idx,
                   struct DeviceParameters *d);

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
                         struct DeviceParameters *d );

#endif   /* ----- #ifndef MODEL_SUPPORT_CUH_  ----- */
/* =============================================================================
 *                         end of file model_support.cuh
 * =============================================================================
 */
