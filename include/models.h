/*
 * =============================================================================
 *
 *       Filename:  models.h
 *
 *    Description:  The header file for all the models
 *
 *        Created:  Fri Aug  7 23:26:29 2015
 *       Modified:  Sat Aug  8 16:12:51 2015
 *
 *         Author:  Huang Zonghao
 *          Email:  coding@huangzonghao.com
 *
 * =============================================================================
 */
#ifndef MODELS_H_
#define MODELS_H_

class CommandQueue;
class SystemInfo;


/* the supported policies */
const int num_policy_options = 3;
const char * policy_options[] = { "all",
                                  "tree",
                                  "fluid" };
bool ModelFluidInit(CommandQueue*, SystemInfo*);
bool ModelFluid(CommandQueue*, SystemInfo*, int idx);
bool ModelDPInit(CommandQueue*, SystemInfo*);
bool ModelDP(CommandQueue*, SystemInfo*, int idx);
#endif   /* ----- #ifndef MODELS_H_  ----- */

/* =============================================================================
 *                         end of file models.h
 * =============================================================================
 */
