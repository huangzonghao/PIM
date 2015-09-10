/*
 * =============================================================================
 *
 *       Filename:  frame.cc
 *
 *    Description:  The definition of the frame function for the PIM problem
 *
 *        Created:  Thu Jul 23 01:09:07 2015
 *       Modified:  Thu 10 Sep 2015 11:51:52 AM HKT
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
#include <cmath>
#include <vector>

#include "../include/command_queue.h"
#include "../include/system_info.h"
#include "../include/models.h"
#include "../include/model_support.h"
#include "../include/cuda_support.h"
#include "../include/device_parameters.h"
#include "../include/demand_distribution.h"

/*
 * ===  FUNCTION  ==============================================================
 *         Name:  LetsRock
 *  Description:  This function controls all the computation resources and ensures
 *                  that the computation is done as requested in the CommandQueue
 *       @param:  CommandQueue and SystemInfo
 *      @return:  whether the operation is suceessful or not
 * =============================================================================
 */
bool LetsRock ( CommandQueue *cmd, SystemInfo *sysinfo, std::vector<float*> host_value_tables ){
    /* to deal with the errors while the computation */
    bool error_msg;

    /* declare the value tables, both host and device */
    float **device_value_tables =
        DeclareValueTable((int)cmd->get_d_param("table_length"), sysinfo);
    float *host_value_table_temp;


    /* check if there are some tasks to recover */

/*-----------------------------------------------------------------------------
 *  the fluid policy
 *-----------------------------------------------------------------------------*/
    if( strcmp(cmd->get_config("policy"), "fluid") == 0 ||
        strcmp(cmd->get_config("policy"),  "all") == 0 ){
        ModelInit(cmd, sysinfo, device_value_tables[0]);
        /* current table is the table to update */
        int current_table_idx = 1;
        int distri_idx = 0;
        for ( int i_period = cmd->get_h_param("T"); i_period > 0; --i_period ){
            if(i_period != 1){
                error_msg = ModelFluid(cmd, sysinfo,
                                       device_value_tables[current_table_idx],
                                       device_value_tables[1 - current_table_idx],
                                       distri_idx,
                                       0);
                current_table_idx = 1 - current_table_idx;
                if(error_msg){
                    printf("Error: Something went wrong in ModelFluid, exit.\n");
                    return false;
                }
            }
            else{
                // first calculate the expect demand for each day
                DemandDistribution *demand = cmd->get_h_demand_pointer(distri_idx);
                float expect_demand = 0;
                for(int i = 0; i < (int)(demand->max_demand - demand->min_demand); ++i){
                    expect_demand += (i + demand->min_demand) * demand->table[i];
                }
                error_msg = ModelFluid(cmd, sysinfo,
                                       device_value_tables[current_table_idx],
                                       device_value_tables[1 - current_table_idx],
                                       distri_idx,
                                       (size_t)ceil(expect_demand));
                if(error_msg){
                    printf("Error: Something went wrong in ModelFluid, exit.\n");
                    return false;
                }
            }
        }
        host_value_table_temp = new float[(int)cmd->get_d_param("table_length")];
        cuda_ReadFromDevice(host_value_table_temp,
                            device_value_tables[1 - current_table_idx],
                            (size_t)cmd->get_d_param("table_length"));
        host_value_tables.push_back(host_value_table_temp);

        printf("Model fluid has finished successfully\n");
    }

/*-----------------------------------------------------------------------------
 *  the dp policy
 *-----------------------------------------------------------------------------*/
    if( strcmp(cmd->get_config("policy"), "DP") == 0 ||
        strcmp(cmd->get_config("policy"),  "all") == 0 ){
        /* for dp policy we need two tabels to store the z, q information on the
         *     device
         */
        int *d_z_records =
            cuda_AllocateMemoryInt((size_t)cmd->get_d_param("table_length"));
        int *d_q_records =
            cuda_AllocateMemoryInt((size_t)cmd->get_d_param("table_length"));

        int current_table_idx = 1;
        int distri_idx = 0;
        ModelInit(cmd, sysinfo, device_value_tables[0]);
        for ( int i_period = cmd->get_h_param("T"); i_period > 0; --i_period ){
            error_msg = ModelDP( cmd,
                                 sysinfo,
                                 device_value_tables[current_table_idx],
                                 device_value_tables[1 - current_table_idx],
                                 distri_idx,
                                 d_z_records, d_q_records);
            current_table_idx = 1 - current_table_idx;
            if(error_msg){
                printf("Error: Something went wrong in ModelDP, exit.\n");
                return false;
            }
        }
        host_value_table_temp = new float[(int)cmd->get_d_param("table_length")];
        cuda_ReadFromDevice(host_value_table_temp,
                            device_value_tables[1 - current_table_idx],
                            (size_t)cmd->get_d_param("table_length"));
        host_value_tables.push_back(host_value_table_temp);

        printf("Model fluid has finished successfully\n");
    }
    return true;
}       /* -----  end of function LetsRock  ----- */


/* =============================================================================
 *                         end of file frame.cc
 * =============================================================================
 */
