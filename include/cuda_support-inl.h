/*
 * =============================================================================
 *
 *       Filename:  cuda_support-inl.h
 *
 *    Description:  This file contains the inline cuda supporting functions
 *
 *        Created:  Fri Jul 24 14:27:18 2015
 *       Modified:  Fri Jul 24 14:27:18 2015
 *
 *         Author:  Huang Zonghao
 *          Email:  coding@huangzonghao.com
 *
 * =============================================================================
 */

/*
 * ===  FUNCTION  ==============================================================
 *         Name:  pass_to_device
 *  Description:  pass the array to device
 *       @param:  pointer to host array, pointer to device array, array size
 *      @return:  void
 * =============================================================================
 */
inline void pass_to_device ( const float* h_array, const float* d_array,\
                             size_t length ){
    checkCudaErrors(cudaMemcpy(d_array, h_array,\
                               length * sizeof(float),\
                               cudaMemcpyHostToDevice));
    return;
}       /* -----  end of function pass_to_device  ----- */
/* reload */
inline void pass_to_device ( const size_t * h_array, const size_t * d_array,\
                             size_t length ){
    checkCudaErrors(cudaMemcpy(d_array, h_array,\
                               length * sizeof(size_t),\
                               cudaMemcpyHostToDevice));
    return;
}       /* -----  end of function pass_to_device  ----- */

/*
 * ===  FUNCTION  ==============================================================
 *         Name:  read_from_device
 *  Description:  read the array from device
 *       @param:  see pass_to_array
 *      @return:  see pass_to_array
 * =============================================================================
 */
inline void read_from_device ( const float* h_array, const float* d_array,\
                               size_t length ){
    checkCudaErrors(cudaMemcpy(h_array, d_array,\
                               length * sizeof(float),\
                               cudaMemcpyDeviceToHost));

    return ;
}       /* -----  end of function read_from_device  ----- */
/* reload */
inline void read_from_device ( const size_t * h_array, const size_t * d_array,\
                               size_t length ){
    checkCudaErrors(cudaMemcpy(h_array, d_array,\
                               length * sizeof(size_t),\
                               cudaMemcpyDeviceToHost));

    return ;
}       /* -----  end of function read_from_device  ----- */

/* =============================================================================
 *                         end of file cuda_support-inl.h
 * =============================================================================
 */
