/*
 * =============================================================================
 *
 *       Filename:  model_support.h
 *
 *    Description:  The header file of the cuda supporting functions related
 *                    to the algorithm
 *
 *        Created:  Sat Aug  8 15:32:25 2015
 *       Modified:  Fri 25 Sep 2015 02:37:32 PM HKT
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

bool model_ValueTableInit(CommandQueue *cmd, SystemInfo *sysinfo, float *value_tables);

float **model_DeclareValueTables(size_t table_length, SystemInfo *sysinfo);
int **model_DeclareMDSpaces(CommandQueue *cmd, SystemInfo *sysinfo);

bool model_CleanUpTables( float **value_tables,
                          size_t num_value_tables,
                          size_t table_length );

#endif   /* ----- #ifndef MODEL_SUPPORT_H_  ----- */
/* =============================================================================
 *                         end of file model_support.h
 * =============================================================================
 */
