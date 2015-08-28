/*
 * =============================================================================
 *
 *       Filename:  models.h
 *
 *    Description:  The header file for all the models
 *
 *        Created:  Fri Aug  7 23:26:29 2015
 *       Modified:  Thu Aug 27 17:41:54 2015
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


/* the supported policies */
const int num_policy_options = 3;
const char *policy_options[] = {  "all",
                                  "tree",
                                  "fluid" };
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
