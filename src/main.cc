/*
 * =============================================================================
 *
 *       Filename:  main.cc
 *
 *    Description:  This file contains the main workflow of the PIM project
 *
 *        Created:  Wed Jul 22 13:57:40 2015
 *       Modified:  Mon Aug 31 22:53:59 2015
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
#include "../include/support.h"
#include "../include/command_queue.h"
#include "../include/system_info.h"
#include "../include/frame.h"
#include "../include/device_parameters.h"

/*
 * ===  FUNCTION  ==============================================================
 *         Name:  main
 *  Description:
 * =============================================================================
 */
int main ( int argc, const char **argv ) {
/*-----------------------------------------------------------------------------
 *  set up the InterruptHandler
 *-----------------------------------------------------------------------------*/
    struct sigaction sigIntHandler;
    sigIntHandler.sa_handler = InterruptHandler;
    sigemptyset(&sigIntHandler.sa_mask);
    sigIntHandler.sa_flags = 0;
    sigaction(SIGINT, &sigIntHandler, NULL);

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
    error_msg = LoadCommands(argc, argv, &cmd);
    if(!error_msg){
        printf("Failure while reading in the commands, exit\n");
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
        printf("Failure while loading the parameters, exit\n");
        return 2;
    }
    cmd.update_device_params();

/*-----------------------------------------------------------------------------
 *  start the main calculation
 *-----------------------------------------------------------------------------*/
    /* start the clock */
    timeval  tv1, tv2;
    /* declare the host value table */
    float *host_value_table = new float[(int)cmd.get_d_param("table_length")];
    gettimeofday(&tv1, NULL);

    error_msg = LetsRock(&cmd, &sysinfo, host_value_table);
    if(!error_msg){
        printf("Something went wrong in LetsRock\n");
        return 3;
    }

    /* end the clock */
    gettimeofday(&tv2, NULL);
    double program_running_time = (double) (tv2.tv_usec - tv1.tv_usec) / 1000000
                                + (double) (tv2.tv_sec - tv1.tv_sec);

    printf("The total time elapsed : %f \n", program_running_time);

    return 0;
}       /* ----------  end of function main  ---------- */


/* =============================================================================
 *                         end of file main.cc
 * =============================================================================
 */
