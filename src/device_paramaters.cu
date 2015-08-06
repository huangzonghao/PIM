/*
 * =============================================================================
 *
 *       Filename:  device_paramaters.cu
 *
 *    Description:  The implementation of DeviceParameters
 *
 *        Created:  Tue Jul 28 14:59:58 2015
 *       Modified:  Thu Aug  6 16:33:27 2015
 *
 *         Author:  Huang Zonghao
 *          Email:  coding@huangzonghao.com
 *
 * =============================================================================
 */
#include <cuda.h>
#include "../thirdparty/nvidia/helper_cuda.h"
#include "../include/host_parameters.h"
#include "../include/device_parameters.h"
/*
 *------------------------------------------------------------------------------
 *       Class:  DeviceParameters
 *      Method:  DeviceParameters
 * Description:  constructor, and will automatically allocate the variable
 *                 spaces in device
 *------------------------------------------------------------------------------
 */

/* :REMARKS:Mon Jul 27 03:52:39 2015:huangzonghao:
 *  we should do mem allocation at the declaration of a deviceparams class
 *  we just want to keep one copy of the device parameters no matter how
 *  many pointers there pointing to that copy.
 *  so only do memalloc when necessary
 */
DeviceParameters::DeviceParameters (const &HostParameters other) {
    /* allocate memory space */
    for (int i = 0; i < num_params_; ++i){
        checkCudaErrors(cudaMalloc(params_ + i, sizeof(float)));
    }
    if ( (int)other["max_demand"] - \
            (int)other["min_demand"] != 0 ){
        float * temp;
        if (demand_distributions.size() != 0){
            demand_distributions.clear();
        }
        for ( int i = 0; i < (int)other["num_distri"]; ++i) {
            checkCudaErrors(cudaMalloc(&temp,\
                        (other["max_demand"] - \
                         other["min_demand"]) * sizeof(float)));
            demand_distributions.push_back(temp);
        }
    }
    /* pass the parameters to the device */
    for (int i = 0; i < num_params_; ++i){
        pass_to_device(other.get_var_ptr("T") + i, params_[i], 1);
    }
    for (int i = 0; i < demand_distributions.size(); ++i){
        pass_to_device(other.get_distribution_ptr(i), demand_distributions[i],\
                other["max_demand"] - other["min_demand"]);
    }

    is_target_set_ = 1;
    is_owner_ = 1;
    /* :REMARKS:Mon Jul 27 03:49:49 2015:huangzonghao:
     *  still cannot allocate the device table here
     */
}  /* -----  end of method DeviceParameters::DeviceParameters  (constructor)  ----- */

/*
 *------------------------------------------------------------------------------
 *       Class:  DeviceParameters
 *      Method:  DeviceParameters
 * Description:  copy constructor
 *------------------------------------------------------------------------------
 */
DeviceParameters::DeviceParameters ( const DeviceParameters &other ) {
    if ( this != &other ) {
        for (int i = 0; i < num_params_; ++i){
            params_[i] = other.params_[i];
        }
        demand_distributions = other.demand_distributions;
    }
    is_target_set_ = 1;
    is_owner_ = 0;
}  /* -----  end of method DeviceParameters::DeviceParameters  (copy constructor)  ----- */

/*
 *------------------------------------------------------------------------------
 *       Class:  DeviceParameters
 *      Method:  ~DeviceParameters
 * Description:  destructor
 *------------------------------------------------------------------------------
 */
/* :TODO:Mon Jul 27 04:19:06 2015:huangzonghao:
 *  think about the race thing when doing destruction!!!!
 */
DeviceParameters::~DeviceParameters () {
    if ( is_owner_ && is_target_set_ ) {
        for (int i = 0; i < num_params_; ++i){
            checkCudaErrors(cudaFree(params_[i]));
        }
        for (int i = 0; demand_distributions.size(); ++i){
            checkCudaErrors(cudaFree(demand_distributions.data()));
        }
    }
}  /* -----  end of method DeviceParameters::~DeviceParameters  (destructor)  ----- */

/*
 *------------------------------------------------------------------------------
 *       Class:  DeviceParameters
 *      Method:  get_var_ptr
 * Description:  return the pointer pointing to the parameters of the
 *                 DeviceParameters. The function is for internal use only
 *------------------------------------------------------------------------------
 */
float ** DeviceParameters::get_var_ptr (const std::string &var) {
    for (int i = 0; i < num_params_; ++i){
        if (var == param_names_[i]){
            return params_ + i;
        }
    }
    printf("Error: DeviceParameters doesn't have %s as a variable, pointer not found.");
    return NULL;
}       /* -----  end of method DeviceParameters::get_var_ptr  ----- */

/*
 *------------------------------------------------------------------------------
 *       Class:  DeviceParameters
 *      Method:  get_float_ptr
 * Description:  return the float pointer pointing to the parameters stored
 *                 in the device. The function is for internal use only
 *------------------------------------------------------------------------------
 */
float * DeviceParameters::get_float_ptr (const std::string &var) {
    if (get_var_ptr(var) == NULL){
        /*
         * note here either a invalid variable name or the device mem hasn't
         *     been allocated
         */
        printf("Error: DeviceParameters::get_float_ptr failed, cannot get pointer.");
        return NULL;
    }
    return *get_var_ptr(var);
}       /* -----  end of method DeviceParameters::get_float_ptr  ----- */

/*
 *------------------------------------------------------------------------------
 *       Class:  DeviceParameters
 *      Method:  operator =
 * Description:  assignment operator
 *------------------------------------------------------------------------------
 */
DeviceParameters&
DeviceParameters::operator = ( const DeviceParameters &other ) {
    if ( this != &other ) {
        for (int i = 0; i < num_params_; ++i){
            params_[i] = other.params_[i];
        }
        demand_distributions = other.demand_distributions;
    }
    is_target_set_ = 1;
    is_owner_ = 0;
    return *this;
}


/* :REMARKS:Mon Jul 27 01:47:27 2015:huangzonghao:
 * Now with the following function, loading the parameters to the device
 * becomes fairly easy, we just need to load to the host first then let the
 * device equal to the host
 */
DeviceParameters&
DeviceParameters::operator = ( const HostParameters &other ) {
    /* first check if the object is capable to hold a device copy */
    if (is_owner_){
        printf( "You cannot assign to a owner of"\
                " a device copy, use the update function\n");
        exit(EXIT_FAILURE);
    }

    /* allocate memory space */
    for (int i = 0; i < num_params_; ++i){
        checkCudaErrors(cudaMalloc(params_ + i, sizeof(float)));
    }
    if ( (int)other["max_demand"] - \
            (int)other["min_demand"] != 0 ){
        float * temp;
        if (demand_distributions.size() != 0){
            demand_distributions.clear();
        }
        for ( int i = 0; i < (int)other["num_distri"]; ++i) {
            checkCudaErrors(cudaMalloc(&temp,\
                        (other["max_demand"] - \
                         other["min_demand"]) * sizeof(float)));
            demand_distributions.push_back(temp);
        }
    }
    /* pass the parameters to the device */
    for (int i = 0; i < num_params_; ++i){
        pass_to_device(other.get_var_ptr("T") + i, params_[i], 1);
    }
    for (int i = 0; i < demand_distributions.size(); ++i){
        pass_to_device(other.get_distribution_ptr(i), demand_distributions[i],\
                other["max_demand"] - other["min_demand"]);
    }

    is_target_set_ = 1;
    is_owner_ = 1;

    return *this;
}
/* -----  end of method DeviceParameters::operator =  (assignment operator)  ----- */



/*
 *------------------------------------------------------------------------------
 *       Class:  DeviceParameters
 *      Method:  is_complete
 * Description:  validation of the completeness of the object -- whether all
 *                 the pointers exist
 *------------------------------------------------------------------------------
 */
bool DeviceParameters::is_complete () {
    for (int i = 0; i < num_params_; ++i){
        if (params_[i] == NULL)
            return false;
    }
    return true;
}       /* -----  end of method DeviceParameters::is_complete  ----- */
/*
 *------------------------------------------------------------------------------
 *       Class:  DeviceParameters
 *      Method:  is_owner
 * Description:  check if the this struct is the owner of some device data
 *                 copy
 *------------------------------------------------------------------------------
 */
bool DeviceParameters::is_owner () {
    return is_owner_;
}       /* -----  end of method DeviceParameters::is_owner  ----- */

/*
 *------------------------------------------------------------------------------
 *       Class:  DeviceParameters
 *      Method:  is_linked
 * Description:  check if the current struct is linked to some device data copy
 *------------------------------------------------------------------------------
 */
bool DeviceParameters::is_linked () {
    return is_target_set_;
}       /* -----  end of method DeviceParameters::is_linked  ----- */


/*
 *------------------------------------------------------------------------------
 *       Class:  DeviceParameters
 *      Method:  set_value
 * Description:  pass the desired value to device, but should be rarely used
 *                 since    the device variables should always be synchorized
 *                 with the host copy
 *------------------------------------------------------------------------------
 */
bool DeviceParameters::set_value ( const std::string &var, float value ) {
    pass_to_device_(&value, get_float_ptr(var), 1);
    return true;
}       /* -----  end of method DeviceParameters::set_value  ----- */



/* =============================================================================
 *                         end of file device_paramaters.cu
 * =============================================================================
 */
