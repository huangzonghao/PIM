/*
 * =============================================================================
 *
 *       Filename:  model_support.h
 *
 *    Description:  The header file of the cuda supporting functions related
 *                    to the algorithm
 *
 *        Created:  Sat Aug  8 15:32:25 2015
 *       Modified:  Thu Aug 27 17:05:53 2015
 *
 *         Author:  Huang Zonghao
 *          Email:  coding@huangzonghao.com
 *
 * =============================================================================
 */
#ifndef MODEL_SUPPORT_H_
#define MODEL_SUPPORT_H_

#include <stdlib.h>

class SystemInfo;
class CommandQueue;

float** DeclareValueTable(size_t table_length, SystemInfo *sysinfo);
bool CleanUpValueTable(float ** &value_tables, size_t table_length);
bool ModelInit(CommandQueue *cmd, SystemInfo *sysinfo, float *value_table);

#endif   /* ----- #ifndef MODEL_SUPPORT_H_  ----- */
/* =============================================================================
 *                         end of file model_support.h
 * =============================================================================
 */
