/*
 * =============================================================================
 *
 *       Filename:  models.h
 *
 *    Description:  The header file for all the models
 *
 *        Created:  Fri Aug  7 23:26:29 2015
 *       Modified:  Thu 10 Sep 2015 04:39:06 AM HKT
 *
 *         Author:  Huang Zonghao
 *          Email:  coding@huangzonghao.com
 *
 * =============================================================================
 */
#ifndef MODELS_H_
#define MODELS_H_
#include <stdlib.h>

class CommandQueue;
class SystemInfo;


/* #ifdef __DEFINE_CONST_VARS_IN_HEADER_ */
/* the supported policies */
static const int num_policy_options = 3;
static const char *policy_options[] = {  "all",
                                  "tree",
                                  "fluid" };
/* #endif */

bool ModelFluid(CommandQueue *cmd,
                SystemInfo *sysinfo,
                float *table_to_update,
                float *table_for_reference,
                int distri_idx,
                size_t depletion_indicator);

bool ModelDP(CommandQueue *cmd,
             SystemInfo *sysinfo,
             float *table_to_update,
             float *table_for_reference,
             int distri_idx,
             int *z, int *q);

#endif   /* ----- #ifndef MODELS_H_  ----- */

/* =============================================================================
 *                         end of file models.h
 * =============================================================================
 */
