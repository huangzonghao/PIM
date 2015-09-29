/*
 * =============================================================================
 *
 *       Filename:  model_support.cu
 *
 *    Description:  The cuda supporting functions related to the algorithm
 *
 *        Created:  Sat Aug  8 15:35:08 2015
 *       Modified:  Tue 29 Sep 2015 03:48:49 PM HKT
 *
 *         Author:  Huang Zonghao
 *          Email:  coding@huangzonghao.com
 *
 * =============================================================================
 */

#include "../include/model_support.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include "../thirdparty/nvidia/helper_cuda.h"
#include "../thirdparty/nvidia/helper_math.h"

#include "../include/cuda_support.cuh"
#include "../include/model_support-inl.cuh"
#include "../include/device_parameters.h"
#include "../include/demand_distribution.h"
#include "../include/command_queue.h"
#include "../include/system_info.h"

/* =============================================================================
 *  The device kernels
 * =========================================================================== */

/*
 * ===  DEVICE KERNEL  =========================================================
 *         Name:  d_DepleteStorage
 *  Description:  deplete the storage by the certain amount
 *       @param:  mD_index, amount to deplete, m
 *      @return:  none
 * =============================================================================
 */
__device__
void d_DepleteStorage(int *mD_index, size_t deplete_amount, size_t m){
    size_t buffer = 0;
    size_t i = 0;
    if (deplete_amount > 0) {
        while(!deplete_amount && i < m){
            if ( !mD_index[i]){
                ++i;
                continue;
            }
            if(mD_index[i] >= deplete_amount )
            {
                mD_index[i] -= deplete_amount;
                deplete_amount  = 0;
                break;
            }
            buffer = deplete_amount - mD_index[i];
            mD_index[i] = 0;
            deplete_amount = buffer;
            buffer = 0;
            ++i;
        }
    }
/* :TODO:Sun Aug  9 09:44:27 2015:huangzonghao:
 *  increase amount
 *  when the deplete_amount is smaller than 0
 */
}       /* -----  end of device kernel d_DepleteStorage  ----- */


/*
 * ===  DEVICE KERNEL  =========================================================
 *         Name:  d_AddStorage
 *  Description:
 *       @param:
 *      @return:  none
 * =============================================================================
 */
/* __device__ */
/* void d_AddStorage(int *md_index, size_t add_amount, size_t m){ */
    /* for(int i = 0; i <  m - 1; ++i){ */
        /* md_index[i] = md_index[i + 1]; */
    /* } */
    /* md_index[m - 1] = add_amount; */
    /* return; */
/* }       [> -----  end of device kernel d_AddStorage  ----- <] */


/*
 * ===  DEVICE KERNEL  =========================================================
 *         Name:  d_GetTomorrowIndex
 *  Description:  calculate the index of Tomorrow's state with today's state
 *                  and today's state change(get from z, q, number of items sold)
 *       @param:  today's state mDarray, today's state change(deplete as positive)
 *                   and m
 *      @return:  the data index of tomorrow's state
 * =============================================================================
 */
__device__
size_t d_GetTomorrowIndex( int *md_space,
                           int today_deplete,
                           int today_demand,
                           int today_order,
                           size_t m){
    d_DepleteStorage(md_space, today_deplete, m);
    /* d_AddStorage(mD_index, today_order, m); */
    md_space[m] = today_order;
    d_DepleteStorage(md_space, today_demand, m + 1);

    return d_check_storage(md_space + 1, m);
}       /* -----  end of device kernel d_GetTomorrowIndex  ----- */

/*
 * ===  DEVICE KERNEL  =========================================================
 *         Name:  d_StateValue
 *  Description:  update one state value given z, q, and the DemandDistribution
 *       @param:  today's mD_index, today's storage, z, q, and parameters
 *      @return:  the expected value of today's storage under certain demand
 * =============================================================================
 */
__device__
float d_StateValue( float *table_for_reference,
                    size_t storage_today,
                    int *my_md_ref,
                    int *my_md_space,
                    int z, int q,
                    int demand_distri_idx,
                    struct DeviceParameters *d ){

    float profit = 0;
    float sum    = 0;
    /* int *mD_temp = new int[d->m]; */
    DemandDistribution demand = *(d->demand_distributions[demand_distri_idx]);
    for ( int i = demand.min_demand; i < demand.max_demand; ++i){
        for (int j = 0; j < d->m; ++j){
            my_md_space[j] = my_md_ref[j];
        }
        profit = d->s * z /* depletion income */
               - d->h * max(int(storage_today) - z , 0) /* holding cost */
               - d->alpha * d->c * q /* ordering cost */
               + d->alpha * d->r * min(i, int(storage_today) - z + q) /* sale income */
               - d->alpha * d->theta * max(my_md_ref[0] - z - i, 0) /* disposal cost */
               /* something wrong here */
               + d->alpha * table_for_reference[d_GetTomorrowIndex(my_md_space, z, i, q, d->m)]; /* value of tomorrow */

        sum += profit * demand.table[i];
    }
    /* delete mD_temp; */
    return sum;
}       /* -----  end of device kernel d_StateValue  ----- */
/*
 * ===  DEVICE KERNEL  =========================================================
 *         Name:  d_StateValueUpdate
 *  Description:  calculate the maximum expected state value for a given state
 *                  with the range of z and q, and also update the value in the
 *                  global value table
 *       @param:  the table to update, the table for reference, the global index
 *                   (dataIdx) of the current state, the range of z and q, and
 *                   the index of the distribution and the DeviceParameters
 *      @return:  none
 * =============================================================================
 */
__device__
void d_justforfun(int c){
    int a  = 1;
    int b = a + 1;
    a = b - a;
    a = c;
    return;
}
__device__
void d_StateValueUpdate( float *table_to_update,
                         float *table_for_reference,
                         size_t dataIdx,
                         int **md_spaces,
                         int *z_records,
                         int *q_records,
                         int min_z, int max_z,
                         int min_q, int max_q,
                         int demand_distri_idx,
                         struct DeviceParameters *d ){

    // Allocate a memory buffer on stack
    // So we don't need to do it for every loop
    // Last dimension are used to store the ordering
    // which could be used for sale
    /* int *mDarray = new int[d->m]; */
    /* int mDarray[2]; */

    size_t my_offset = blockIdx.x * blockDim.x + threadIdx.x;
    int *my_md_ref = md_spaces[0] + my_offset * d->m;
    int *my_md_space = md_spaces[1] + my_offset * (d->m + 1);

    /* temp_z and temp_q are to store the best z and q, for future use */
    int   temp_z         = 0;
    int   temp_q         = 0;
    float max_value      = 0.0;
    float expected_value = 0.0;
    size_t storage_today = d_decode(dataIdx, d->m, d->k, my_md_ref);

    for (int i_z = min_z; i_z <= max_z; ++i_z) {
        for (int i_q = min_q; i_q <= max_q; ++i_q) {
            expected_value = d_StateValue(table_for_reference,
                                          storage_today,
                                          my_md_ref,
                                          my_md_space,
                                          i_z, i_q,
                                          demand_distri_idx,
                                          d);

            // Simply taking the moving maximum
            if (expected_value > max_value + 1e-6) {
                max_value = expected_value;
                temp_z = i_z;
                temp_q = i_q;
            }
        }
    }

    // Store the optimal point and value
    /* if(threadIdx.x < 100){ */
        /* printf("max value : %f\n", max_value); */
    /* } */
    table_to_update[dataIdx] = max_value;

    if(z_records != NULL)
        z_records[dataIdx] = temp_z;
    if(q_records != NULL)
        q_records[dataIdx] = temp_q;

    /* delete mDarray; */
}    /* -----  end of device kernel d_StateValueUpdate  ----- */

/* =============================================================================
 *  The global kernels
 * =========================================================================== */

/*
 * ===  GLOBAL KERNEL  =========================================================
 *         Name:  g_ModelInit
 *  Description:  the kernel to init the value table.
 *       @param:  the DeviceParameters and the pointer to the value table
 * =============================================================================
 */
__global__
void g_ModelInit(struct DeviceParameters d, float *value_table){
    // the total number of threads which have been assigned for this task,
    // oneD layout everywhere
    size_t step_size = gridDim.x * blockDim.x;
    size_t myStartIdx = blockIdx.x * blockDim.x + threadIdx.x;

    for (size_t i = myStartIdx; i < d.table_length; i += step_size){
        value_table[i] = d_check_storage(i, d.m, d.k) * d.s;
    }
    __syncthreads();
    return;
}       /* -----  end of global kernel g_ModelInit  ----- */
/* =============================================================================
 *  The host functions
 * =========================================================================== */

/*
 * ===  FUNCTION  ==============================================================
 *         Name:  model_ValueTableInit
 *  Description:  to initialize the value table at the beginning of the program
 *                  for all policies, we just sell out all the items with the
 *                  salvage price
 *       @param:  the CommandQueue, the SystemInfo, and the pointer to the value
 *                   table
 *      @return:  success or not
 * =============================================================================
 */
bool model_ValueTableInit(CommandQueue *cmd, SystemInfo *sysinfo, float *value_table){
    g_ModelInit<<<sysinfo->get_value("num_cores"), sysinfo->get_value("core_size")>>>
                        (cmd->get_device_param_struct(), value_table);
    return true;
}       /* -----  end of function model_ValueTableInit  ----- */

/*
 * ===  FUNCTION  ==============================================================
 *         Name:  model_DeclareValueTables
 *  Description:  declare the two value tables in the cuda device
 *       @param:  table length
 *      @return:  float**, the pointer returned can be dereferenced on host
 * =============================================================================
 */
float **model_DeclareValueTables(size_t table_length, SystemInfo *sysinfo){
    /* first declare the host space for the two pointers holding the two tables */
    float **temp = new float*[2];
    /* then the device memory space */
    for (int i = 0; i < 2; ++i){
        checkCudaErrors(cudaMalloc(temp + i, table_length * sizeof(float)));
        /* then zeroize the table */
        g_ZeroizeMemory
            <<<sysinfo->get_value("num_cores"), sysinfo->get_value("core_size")>>>
            (temp[i], table_length);
    }
    return temp;
}       /* -----  end of function model_DeclareValueTables  ----- */


/*
 * ===  FUNCTION  ==============================================================
 *         Name:  model_DeclareMDSpaces
 *  Description:  declare the m dimensional lists which will be used as the
 *                  working spaces in kernel functions in cuda device
 *                  the length of each list will depend on the cuda device
 *       @param:  table length
 *      @return:  int**, the pointer returned can be dereferenced on device
 *                  note the length of the table for reference and the length of
 *                  the temp working space are differnt. the pointer to the ref
 *                  table will come first
 * =============================================================================
 */
int **model_DeclareMDSpaces(CommandQueue *cmd, SystemInfo *sysinfo){
    /* first allocate device spaces to hold the two pointers,
     * this will be the returned value
     */
    int **d_pointers = NULL;
    checkCudaErrors(cudaMalloc(&d_pointers, 2 * sizeof(int*)));

    /* then some spaces on host for intermediate process */
    int **h_pointers = new int*[2];

    /* in order to make the two tables adjunct, we first declare one long table and
     * then assign another pointer to the middle point
     */
    size_t ref_table_length = sysinfo->get_value("num_cores")
                               * sysinfo->get_value("core_size")
                               * cmd->get_h_param("m");

    size_t space_table_length = sysinfo->get_value("num_cores")
                               * sysinfo->get_value("core_size")
                               * (cmd->get_h_param("m") + 1);

    int *temp_pointer = NULL;
    checkCudaErrors(cudaMalloc(&temp_pointer,
                    (ref_table_length + space_table_length) * sizeof(int)));
    h_pointers[0] = temp_pointer;

    /* then zeroize the table */
    g_ZeroizeMemory
        <<<sysinfo->get_value("num_cores"), sysinfo->get_value("core_size")>>>
        (h_pointers[0], ref_table_length + space_table_length );

    h_pointers[1] = h_pointers[0] + ref_table_length;

    checkCudaErrors(cudaMemcpy(d_pointers, h_pointers,
                               2 * sizeof(int*),
                               cudaMemcpyHostToDevice));

    delete h_pointers;

    return d_pointers;
}       /* -----  end of function model_DeclareMDSpaces  ----- */
/*
 * ===  FUNCTION  ==============================================================
 *         Name:  model_CleanUpTables
 *  Description:  clean up the memory space declared in DeclareValueTable
 *                  both host memory and the device memory
 *       @param:  float** and table length
 *      @return:  success or not
 * =============================================================================
 */
bool model_CleanUpTables( float **value_tables,
                          size_t num_value_tables,
                          size_t table_length ){
    /* first the device memory space */
    for (int i = 0; i < num_value_tables; ++i){
        checkCudaErrors(cudaFree(value_tables[i]));
    }
    /* then the host memory */
    delete value_tables;
    value_tables = NULL;
    return true;
}       /* -----  end of function model_CleanUpTables  ----- */


/* =============================================================================
 *                         end of file model_support.cu
 * =============================================================================
 */
