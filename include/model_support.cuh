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
#ifndef MODEL_SUPPORT_H_
#define MODEL_SUPPORT_H_
__device__
void d_DepleteStorage(int * mD_index, size_t deplete_amount, size_t m);

__device__
size_t d_GetTomorrowIndex(int * mD_index, int today_deplete, size_t m);

__device__
float d_StateValue(float * last_table,
                   int * mD_index,
                   size_t storage_today,
                   int z,
                   int q,
                   struct DeviceParameters &d,
                   int demand_table_idx);

#endif   /* ----- #ifndef MODEL_SUPPORT_H_  ----- */
/* =============================================================================
 *                         end of file model_support.cuh
 * =============================================================================
 */
