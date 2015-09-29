/*
 * =============================================================================
 *
 *       Filename:  frame.cc
 *
 *    Description:  The definition of the frame function for the PIM problem
 *
 *        Created:  Thu Jul 23 01:09:07 2015
 *       Modified:  Tue 29 Sep 2015 05:55:44 PM HKT
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
#include "../include/support.h"

/*
 * ===  FUNCTION  ==============================================================
 *         Name:  LetsRock
 *  Description:  This function controls all the computation resources and ensures
 *                  that the computation is done as requested in the CommandQueue
 *       @param:  CommandQueue and SystemInfo
 *      @return:  whether the operation is suceessful or not
 * =============================================================================
 */
bool LetsRock ( CommandQueue *cmd,
                SystemInfo *sysinfo,
                std::vector<float*> &host_value_tables,
                std::vector<int*> &optimal_zq ){
    /* to deal with the errors while the computation */
    bool error_msg;

    /* declare the value tables, both host and device */
    printf("allocating the value table \n");
    float **device_value_tables =
        model_DeclareValueTables((size_t)cmd->get_d_param("table_length"), sysinfo);
    float *host_value_table_temp = NULL;

    int *host_q_records_temp = NULL;
    int *host_z_records_temp = NULL;

    int *device_z_records =
        cuda_AllocateMemoryInt((size_t)cmd->get_d_param("table_length"));
    int *device_q_records =
        cuda_AllocateMemoryInt((size_t)cmd->get_d_param("table_length"));

    /* since the in-kernel dynamic allocation is barely can be useful, we allocate
     * some global memory to do the work
     */
    /* the first list is for reference and the second one is the working space */
    int **device_md_spaces = model_DeclareMDSpaces(cmd, sysinfo);


    /* check if there are some tasks to recover */

    int num_periods_to_run = cmd->get_h_param("T");
    /* if(cmd->check_command("recovery")){ */
        /* float *r_table_to_update = new float[(int)cmd->get_d_param("table_length")]; */
        /* float *r_table_for_reference = new float[(int)cmd->get_d_param("table_length")]; */
        /* LoadProgress( cmd, */
                      /* set_period, */
                      /* r_table_to_update, */
                      /* r_table_for_reference); */
    /* } */
/*-----------------------------------------------------------------------------
 *  the fluid policy
 *-----------------------------------------------------------------------------*/
    if( strcmp(cmd->get_config("policy"), "fluid") == 0 ||
        strcmp(cmd->get_config("policy"),  "all") == 0 ){
        if(cmd->check_command("recovery")){

        }
        model_ValueTableInit(cmd, sysinfo, device_value_tables[0]);
        /* current table is the table to update */
        int current_table_idx = 1;
        int distri_idx = 0;
        for ( int i_period = num_periods_to_run; i_period > 0; --i_period ){
            error_msg = ModelFluid( cmd, sysinfo,
                                    device_value_tables[current_table_idx],
                                    device_value_tables[1 - current_table_idx],
                                    distri_idx,
                                    i_period,
                                    device_md_spaces,
                                    device_z_records,
                                    device_q_records );

            current_table_idx = 1 - current_table_idx;
            if(!error_msg){
                printf("Error: Something went wrong in ModelFluid, "
                        "period %d. Exit.\n", i_period);
                return false;
            }
        }
        host_value_table_temp = new float[(int)cmd->get_d_param("table_length")];
        cuda_ReadFromDevice(host_value_table_temp,
                            device_value_tables[1 - current_table_idx],
                            (size_t)cmd->get_d_param("table_length"));

        host_value_tables.push_back(host_value_table_temp);

        host_z_records_temp = new int[(int)cmd->get_d_param("table_length")];
        cuda_ReadFromDevice(host_z_records_temp,
                            device_z_records,
                            (size_t)cmd->get_d_param("table_length"));
        host_q_records_temp = new int[(int)cmd->get_d_param("table_length")];
        cuda_ReadFromDevice(host_q_records_temp,
                            device_q_records,
                            (size_t)cmd->get_d_param("table_length"));

        optimal_zq.push_back(host_z_records_temp);
        optimal_zq.push_back(host_q_records_temp);

        printf("Model fluid has finished successfully\n");
    }

/*-----------------------------------------------------------------------------
 *  the dp policy
 *-----------------------------------------------------------------------------*/
    if( strcmp(cmd->get_config("policy"), "tree") == 0 ||
        strcmp(cmd->get_config("policy"),  "all") == 0 ){


        int current_table_idx = 1;
        int distri_idx = 0;
        model_ValueTableInit(cmd, sysinfo, device_value_tables[0]);

        for ( int i_period = num_periods_to_run; i_period > 0; --i_period ){
            error_msg = ModelDP( cmd,
                                 sysinfo,
                                 device_value_tables[current_table_idx],
                                 device_value_tables[1 - current_table_idx],
                                 distri_idx,
                                 i_period,
                                 device_md_spaces,
                                 device_z_records, device_q_records);

            current_table_idx = 1 - current_table_idx;
            if(!error_msg){
                printf("Error: Something went wrong in ModelDP, "
                       "period %d. Exit.\n", i_period);
                return false;
            }
        }
        host_value_table_temp = new float[(int)cmd->get_d_param("table_length")];
        cuda_ReadFromDevice(host_value_table_temp,
                            device_value_tables[1 - current_table_idx],
                            (size_t)cmd->get_d_param("table_length"));
        host_value_tables.push_back(host_value_table_temp);

        host_z_records_temp = new int[(int)cmd->get_d_param("table_length")];
        cuda_ReadFromDevice(host_z_records_temp,
                            device_z_records,
                            (size_t)cmd->get_d_param("table_length"));
        host_q_records_temp = new int[(int)cmd->get_d_param("table_length")];
        cuda_ReadFromDevice(host_q_records_temp,
                            device_q_records,
                            (size_t)cmd->get_d_param("table_length"));

        optimal_zq.push_back(host_z_records_temp);
        optimal_zq.push_back(host_q_records_temp);

        printf("Model DP has finished successfully\n");
    }

    /* we need to clean up the value tables we declared here before return */

    return true;
}       /* -----  end of function LetsRock  ----- */


/* =============================================================================
 *                         end of file frame.cc
 * =============================================================================
 */
