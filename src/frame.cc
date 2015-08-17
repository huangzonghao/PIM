/*
 * =============================================================================
 *
 *       Filename:  frame.cc
 *
 *    Description:  The definition of the frame function for the PIM problem
 *
 *        Created:  Thu Jul 23 01:09:07 2015
 *       Modified:  Mon Aug 10 23:17:03 2015
 *
 *         Author:  Huang Zonghao
 *          Email:  coding@huangzonghao.com
 *
 * =============================================================================
 */

/* :REMARKS:Fri Aug  7 23:29:31 2015:huangzonghao:
 *  Note for simplicity, the model function will only take care of one period. i
 *  The for loop will be called by the LetsRock function. Therefore we can
 *  implement all kinds of record and recovery thing within the LetsRock function
 */

#include "../include/frame.h"

#include <cstring>

#include "../include/command_queue.h"
#include "../include/system_info.h"
#include "../include/models.h"
#include "../include/model_support.h"
#include "../include/device_parameters.h"

/*
 * ===  FUNCTION  ==============================================================
 *         Name:  LetsRock
 *  Description:  This function controls all the computation resources and ensures
 *                  that the computation is done as requested in the CommandQueue
 *       @param:  CommandQueue and SystemInfo
 *      @return:  whether the operation is suceessful or not
 * =============================================================================
 */
bool LetsRock ( CommandQueue * cmd, SystemInfo * sysinfo ){
    /* first do some command initialization such as declare two value tables */
    float **value_tables =\
        DeclareValueTable(cmd->get_device_param_pointer()->table_length, sysinfo);


    /* check if there are some tasks to recover */

/*-----------------------------------------------------------------------------
 *  the fluid policy
 *-----------------------------------------------------------------------------*/
    if( strcmp(cmd->get_config("policy"), "fluid") == 0 ||
        strcmp(cmd->get_config("policy"),  "all") == 0 ){
        ModelFluidInit(cmd, sysinfo, value_tables);
        for ( int i_period = cmd->get_h_param("T"); i_period > 0; --i_period ){
            ModelFluid(cmd, sysinfo, i_period);
        }
    }
/*-----------------------------------------------------------------------------
 *  the dp policy
 *-----------------------------------------------------------------------------*/
    if( strcmp(cmd->get_config("policy"), "DP") == 0 ||
        strcmp(cmd->get_config("policy"),  "all") == 0 ){
        ModelDPInit(cmd, sysinfo);
        for ( int i_period = cmd->get_h_param("T"); i_period > 0; --i_period ){
            ModelDP(cmd, sysinfo, i_period);
        }
    }
    return true;
}       /* -----  end of function LetsRock  ----- */


/* =============================================================================
 *                         end of file frame.cc
 * =============================================================================
 */
