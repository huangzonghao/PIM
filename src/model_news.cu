/*
 * =============================================================================
 *
 *       Filename:  model_dp.cu
 *
 *    Description:  All the functions to compute with the dynamic programming
 *                    algorithm
 *
 *        Created:  Fri Aug  7 23:47:24 2015
 *       Modified:  Sat Aug  8 12:16:58 2015
 *
 *         Author:  Huang Zonghao
 *          Email:  coding@huangzonghao.com
 *
 * =============================================================================
 */
#include "../include/models.h"

 // Helper function to get CUDA thread id
// whenever we use __device__ function
__device__ inline size_t
get_thread_id() {

    size_t blockId = blockIdx.x +
                     blockIdx.y * gridDim.x +
                     gridDim.x * gridDim.y * blockIdx.z;
    return blockId * blockDim.x + threadIdx.x;
}


// Using these values for general CUDA GPU is just fine
inline void
get_grid_dim(dim3* block_dim, dim3* grid_dim, size_t batch_size) {

    size_t n_block = batch_size / 512 + 1;

    assert(block_dim && grid_dim);
    *block_dim = dim3(512, 1, 1);
    *grid_dim = dim3(4096, n_block / 4096 + 1, 1);
}


// CUDA Kernel function for initialization
__global__ void
init_kernel(float *current_values,
            size_t batch_idx,
            size_t batch_size) {

    size_t thread_idx = get_thread_id();

    if (thread_idx < batch_size) {

        size_t current = batch_idx * batch_size + thread_idx;
        size_t parent = current - batch_size;

        if (current == 0) {
            current_values[current] = 0.0;
        } else {
            current_values[current] = current_values[parent] + 1.0;
        }
    }
}


// Plain C function for interact with kernel
void
init_states(float *current_values) {

    size_t num_states = std::pow(n_capacity, n_dimension);

    // The very first state
    init_kernel<<<1, 1>>>(current_values, 0, 1);

    for (size_t d = 0; d < n_dimension; d++) {

        size_t batch_size = pow(n_capacity, d);

        dim3 block_dim, grid_dim;
        get_grid_dim(&block_dim, &grid_dim, batch_size);

        for (size_t batch_idx = 1; batch_idx < n_capacity; batch_idx++) {
            init_kernel<<<grid_dim, block_dim>>>(current_values,
                                                 batch_idx,
                                                 batch_size);
        }
    }

    cudaDeviceSynchronize();
    cudaThreadSynchronize();

}

// The CUDA kernel function for DP_news
__global__ void
iter_kernel(float *current_values,
            dp_int *depletion,
            dp_int *order,
            float *future_values,
            int period,
            size_t batch_idx,
            size_t batch_size) {

    size_t thread_idx = get_thread_id();

    if (thread_idx < batch_size) {
       // first update current_values

        size_t current = batch_idx * batch_size + thread_idx;
        size_t parent = current - batch_size;

        int state[n_dimension+1] = {};
        decode(state, current);
        int currentsum = sum(state, n_dimension+1);
        int n_depletion = 0;
        int n_order = 0;

        float max_value = 0.0;
    
        struct Demand demand = demand_distribution_at_period[0];
     
        // Case 1: period < T-L-1;
        if (period < n_period- n_dimension){
               n_depletion= 0;
               n_order =0;
               if (n_capacity-1- currentsum >0){
                  n_order= n_capacity-1 - currentsum;
               }
               current_values[current] = stateValue(current, n_depletion, n_order, future_values,demand, period);  
               depletion[current] = (dp_int) n_depletion;
               order[current] = (dp_int) n_order;
        }
        // Case 2
        else {
           if (current==0 || depletion[parent]== 0){
              for (int i = 0; i <= 1; i++){        
                   int j= 0;
                   if (currentsum- i < n_capacity-1){
                      j = n_capacity-1- currentsum + i;
                   }
                   float expected_value = stateValue(current,i,j,future_values,demand, period) ;
                
                 // Simply taking the moving maximum
                   if (expected_value > max_value + 1e-8) {
                       max_value = expected_value;
                       n_depletion = i;
                       n_order = j;
                    }
               }   
               current_values[current] = max_value;
               depletion[current] = (dp_int) n_depletion;
               order[current] = (dp_int) n_order;
           }
          else{
              current_values[current] = stateValue(current,depletion[parent]+1, order[parent], future_values,demand, period);  
              depletion[current]= depletion[parent]+1;
              order[current]= order[parent];
          }
      }  
   }
}

// Plain C function to interact with kernel
// The structure is essentially the same as init_states.
// If you feel confused, start from there!
void
iter_states(float *current_values,
            dp_int *depletion,
            dp_int *order,
            float *future_values,
            int period) {

    size_t num_states = std::pow(n_capacity, n_dimension);

    // The very first state 0,0,...,0
    iter_kernel<<<1, 1>>>(current_values,
                          depletion,
                          order,
                          future_values,
                          period,
                          0, 1);

    for (size_t d = 0; d < n_dimension; d++) {

        size_t batch_size = pow(n_capacity, d);

        dim3 block_dim, grid_dim;
        get_grid_dim(&block_dim, &grid_dim, batch_size);

        for (size_t batch_idx = 1; batch_idx < n_capacity; batch_idx++) {
            iter_kernel<<<grid_dim, block_dim>>>(current_values,
                                                 depletion,
                                                 order,
                                                 future_values,
                                                 period,
                                                 batch_idx,
                                                 batch_size);
        }
    }

    cudaDeviceSynchronize();
    cudaThreadSynchronize();

}

/*
 * ===  FUNCTION  ==============================================================
 *         Name:  ModelDPInit
 *  Description:  The initialization function for ModelDP
 *       @param:  the control sequence, the system information
 *      @return:  success or not
 * =============================================================================
 */
bool ModelDPInit(CommandQueue * cmd, SystemInfo * sysinfo){
    g_ModelFluidInit<<<sysinfo->get_value["num_cores"],\
                        sysinfo->get_value["core_size"]>>>\
                        (*(cmd->get_device_param_pointer), value_table);
    return true;
}       /* -----  end of function ModelDPInit  ----- */

/* :REMARKS:Sat Aug  8 12:16:39 2015:huangzonghao:
 *  shouldn't the number of index be stored in the CommandQueue????
 */
/*
 * ===  FUNCTION  ==============================================================
 *         Name:  ModelDP
 *  Description:  to update the table for one period with the dynamic programming
 *                  algorithm
 *       @param:  control sequence, system information, the index of the period
 *      @return:  success or not
 * =============================================================================
 */
bool ModelDP(CommandQueue * cmd, SystemInfo * sysinfo, int idx){

    size_t num_states = std::pow(n_capacity, n_dimension);

    float *h_current_values;
    float *h_future_values;
    dp_int *h_depletion;
    dp_int *h_order;

    checkCudaErrors(cudaHostAlloc((void **)&h_current_values,
                                  sizeof(float) * num_states,
                                  cudaHostAllocMapped));
    checkCudaErrors(cudaHostAlloc((void **)&h_future_values,
                                  sizeof(float) * num_states,
                                  cudaHostAllocMapped));
    checkCudaErrors(cudaHostAlloc((void **)&h_depletion,
                                  sizeof(dp_int) * num_states,
                                  cudaHostAllocMapped));
    checkCudaErrors(cudaHostAlloc((void **)&h_order,
                                  sizeof(dp_int) * num_states,
                                  cudaHostAllocMapped));

    float *d_current_values;
    float *d_future_values;
    dp_int *d_depletion;
    dp_int *d_order;

    cudaSetDeviceFlags(cudaDeviceMapHost);



    checkCudaErrors(cudaHostGetDevicePointer((void **)&d_current_values,
                                             (void *)h_current_values, 0));
    checkCudaErrors(cudaHostGetDevicePointer((void **)&d_future_values,
                                             (void *)h_future_values, 0));
    checkCudaErrors(cudaHostGetDevicePointer((void **)&d_depletion,
                                             (void *)h_depletion, 0));
    checkCudaErrors(cudaHostGetDevicePointer((void **)&d_order,
                                             (void *)h_order, 0));

 /*   FILE *fp;
    fp = fopen("/ghome/hzhangaq/DP-parellel-computing/CCode/dp0701.log","r");
    for (int i=0; i <  num_states; i++){
        fscanf(fp,"%f", &h_future_values[i]);
    } */
    init_states(d_future_values);

  //  std::cout << "depletion,order,value" << std::endl;

    for (int period = 0; period < n_period; period++) {

        iter_states(d_current_values,
                    d_depletion,
                    d_order,
                    d_future_values,
                    period);

        // Print the results
                float *tmp = d_future_values;
        d_future_values = d_current_values;
        d_current_values = tmp;
    }
   //int state[n_dimension+1] = {};
   for (int idx = 0; idx < num_states; idx++) {
      int idxsum= 0;
      int idx_1 = idx;
      for (int i= n_dimension-1; i>= 0; i--){
          idxsum += idx_1 % n_capacity;
          idx_1 /= n_capacity;
      }
      if (idxsum <= cvalue){
         /*   int exp = std::pow(n_capacity, n_dimension-1);
            int i = idx;
            for (int k = 0; k < n_dimension; k++) {
                if (k > 0) {
                    std::cout << ',';
                }
                std::cout << i / exp;
                i %= exp;
                exp /= n_capacity;
            }
            std::cout << '\t';
            std::cout << static_cast<int>(d_depletion[idx]) << ',';
            std::cout << static_cast<int>(d_order[idx]) << ',';  */
            std::cout << std::fixed << std::setprecision(4) << d_future_values[idx];
            std::cout << '\n';
      }
    }
        std::cout << std::endl;



    checkCudaErrors(cudaFreeHost((void *)h_current_values));
    checkCudaErrors(cudaFreeHost((void *)h_future_values));
    checkCudaErrors(cudaFreeHost((void *)h_depletion));
    checkCudaErrors(cudaFreeHost((void *)h_order));

    return 0;
}       /* -----  end of function ModelDP  ----- */


/* =============================================================================
 *                         end of file model_dp.cu
 * =============================================================================
 */
