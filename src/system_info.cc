/*
 * =============================================================================
 *
 *       Filename:  system_info.cc
 *
 *    Description:   Implementation of SystemInfo
 *
 *        Created:  Tue Jul 28 14:49:25 2015
 *       Modified:  Thu 10 Sep 2015 04:13:28 AM HKT
 *
 *         Author:  Huang Zonghao
 *          Email:  coding@huangzonghao.com
 *
 * =============================================================================
 */
#include "../include/system_info.h"

#include <cstring>
#include <stdio.h>

#include "../include/cuda_support.h"

/*
 *------------------------------------------------------------------------------
 *       Class:  SystemInfo
 *      Method:  get_value
 * Description:  return the system configuration values
 *------------------------------------------------------------------------------
 */
int SystemInfo::get_value (const char* var) {
    if(strcmp(var, "num_devs") == 0){
        return num_cuda_devices_;
    }

    if(strcmp(var, "num_cores") == 0){
        return num_cuda_cores_;
    }

    if(strcmp(var, "core_size") == 0){
        return cuda_core_size_;
    }

    printf("Invalid SystemInfo variable name, exit.");
    exit(-1);
    return -1;
}       /* -----  end of method SystemInfo::get_value  ----- */


/*
 *------------------------------------------------------------------------------
 *       Class:  SystemInfo
 *      Method:  print_sys_info
 * Description:  print out the SystemInfo
 *------------------------------------------------------------------------------
 */
void SystemInfo::print_sys_info () {
    printf( "System Configuration : "
            "   Number of CUDA Devices : \e[38;5;166m%d\e[m"
            "   Number of cores : \e[38;5;166m%d\e[m"
            "   Number of threads per core : \e[38;5;166m%d\e[m",
            num_cuda_devices_, num_cuda_cores_, cuda_core_size_);
    return ;
}       /* -----  end of method SystemInfo::print_sys_info  ----- */

/*
 *------------------------------------------------------------------------------
 *       Class:  SystemInfo
 *      Method:  check_gpu
 * Description:  check and update the SystemInfo on GPU
 *------------------------------------------------------------------------------
 */
void SystemInfo::check_gpu () {
    cuda_CheckGPU(&num_cuda_devices_, &num_cuda_cores_, &cuda_core_size_);
    return ;
}       /* -----  end of method SystemInfo::check_gpu  ----- */



/* =============================================================================
 *                         end of file system_info.cc
 * =============================================================================
 */
