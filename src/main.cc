/*
 * =============================================================================
 *
 *       Filename:  main.cc
 *
 *    Description:  This file contains the main workflow of the PIM project
 *
 *        Created:  Wed Jul 22 13:57:40 2015
 *       Modified:  Thu 24 Sep 2015 09:02:27 AM HKT
 *
 *         Author:  Huang Zonghao
 *          Email:  coding@huangzonghao.com
 *
 * =============================================================================
 */


/* #####   HEADER FILE INCLUDES   ############################################ */
#include <sys/time.h>
#include <stdlib.h>
#include <unistd.h>
#include <signal.h>
#include <stdio.h>
#include <vector>
#include <iostream>

#include "../include/support.h"
#include "../include/support-inl.h"
#include "../include/command_queue.h"
#include "../include/system_info.h"
#include "../include/frame.h"
#include "../include/device_parameters.h"
/*
 * ===  FUNCTION  ==============================================================
 *         Name:  main
 * =============================================================================
 */
int main (int argc, const char **argv) {
    printf("Program started\n"
            "The current workding directory is %s\n\n",
            ExeCMD("pwd"));
/*-----------------------------------------------------------------------------
 *  set up the InterruptHandler
 *-----------------------------------------------------------------------------*/
    /* struct sigaction sigIntHandler;
     * sigIntHandler.sa_handler = InterruptHandler;
     * sigemptyset(&sigIntHandler.sa_mask);
     * sigIntHandler.sa_flags = 0;
     * sigaction(SIGINT, &sigIntHandler, NULL);
     */

/*-----------------------------------------------------------------------------
 *  declare the administrative variables
 *-----------------------------------------------------------------------------*/
    CommandQueue cmd;
    bool error_msg = false;
    SystemInfo sysinfo;
    sysinfo.check_gpu();

/*-----------------------------------------------------------------------------
 *  load the system commands
 *-----------------------------------------------------------------------------*/
    error_msg = LoadCommands(argc, (char**)argv, &cmd);
    if(!error_msg){
        printf("Error detected while reading in the commands, exit\n");
        return 1;
    }

    /* printusage has the early exit, so check it first */
    if(cmd.check_command("print_help")){
        PrintUsage();
        return 0;
    }

/*-----------------------------------------------------------------------------
 *  load the parameters
 *-----------------------------------------------------------------------------*/
    error_msg = LoadParameters(&cmd);
    if(!error_msg){
        printf("Error detected while loading the parameters, exit\n");
        return 2;
    }
    cmd.update_device_params();

/*-----------------------------------------------------------------------------
 *  start the main calculation
 *-----------------------------------------------------------------------------*/
    /* start the clock */
    timeval  tv1, tv2;
    /* declare the host value table */
    std::vector<float*> host_value_tables;
    /* the container will be filled in the frame function
     *     and the size of the vector will be decided by the number of policies
     *     enabled
     */

    gettimeofday(&tv1, NULL);

    error_msg = LetsRock(&cmd, &sysinfo, host_value_tables);
    if(!error_msg){
        printf("Something went wrong in LetsRock\n");
        return 3;
    }
    printf("main done\n");

    /* end the clock */
    gettimeofday(&tv2, NULL);
    double program_running_time = (double) (tv2.tv_usec - tv1.tv_usec) / 1000000
                                + (double) (tv2.tv_sec - tv1.tv_sec);

/*-----------------------------------------------------------------------------
 *  some post process
 *-----------------------------------------------------------------------------*/

    /* 1. write the output
     * 2. print the status of the task
     */
    printf("The total time elapsed : %f \n", program_running_time);
    error_msg = WriteOutputFile(host_value_tables[0],
                                cmd.get_d_param("table_length"),
                                1,//output format
                                cmd.get_config("output_file_name"));

    printf("Success: the program finished successfully!\n");
    return 0;
}       /* ----------  end of function main  ---------- */


/* =============================================================================
 *                         end of file main.cc
 * =============================================================================
 */
