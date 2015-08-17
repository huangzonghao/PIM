/*
 * =============================================================================
 *
 *       Filename:  system_info.cc
 *
 *    Description:   Implementation of SystemInfo
 *
 *        Created:  Tue Jul 28 14:49:25 2015
 *       Modified:  Sun Aug  9 15:15:23 2015
 *
 *         Author:  Huang Zonghao
 *          Email:  coding@huangzonghao.com
 *
 * =============================================================================
 */
#include "../include/system_info.h"

#include "../include/cuda_support.h"

/*
 *------------------------------------------------------------------------------
 *       Class:  SystemInfo
 *      Method:  get_value
 * Description:  return the system configuration values
 *------------------------------------------------------------------------------
 */
int SystemInfo::get_value (const char* var) {
switch (var) {
    case "num_devs":
        return num_cuda_devices_;
        break;

    case "num_cores":
        return num_cuda_cores_;
        break;

    case "core_size":
        return cuda_core_size_;
        break;

    default:
        printf("Invalid SystemInfo variable name, exit.");
        exit(-1);
        break;
}            /* -----  end switch  ----- */
    return ;
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
            "   Number of CUDA Devices : " << "\e[38;5;166m%d\e[m"
            "   Number of cores : " << "\e[38;5;166m%d\e[m"
            "   Number of threads per core : " << "\e[38;5;166m%d\e[m",
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
