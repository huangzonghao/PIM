/*
 * =============================================================================
 *
 *       Filename:  host_parameters.cc
 *
 *    Description:  The implementation of HostParameters
 *
 *        Created:  Tue Jul 28 14:58:27 2015
 *       Modified:  Fri Aug  7 17:05:44 2015
 *
 *         Author:  Huang Zonghao
 *          Email:  coding@huangzonghao.com
 *
 * =============================================================================
 */
#include "host_parameters.h"
#include "device_parameters.h"
/* =============================================================================
 *                  Methods of HostParameters
 * =========================================================================== */

/*
 *------------------------------------------------------------------------------
 *       Class:  HostParameters
 *      Method:  HostParameters
 * Description:  constructor
 *------------------------------------------------------------------------------
 */
HostParameters::HostParameters () {
    for (int i = 0; i < num_params_; ++i){
        params_[i] = 0;
    }
}  /* -----  end of method HostParameters::HostParameters  (constructor)  ----- */

/*
 *------------------------------------------------------------------------------
 *       Class:  HostParameters
 *      Method:  HostParameters
 * Description:  copy constructor
 *------------------------------------------------------------------------------
 */
HostParameters::HostParameters ( const HostParameters &other ) {
    if (this != &other) {
        for (int i = 0; i < num_params_; ++i){
            params_[i] = other.params_[i];
        }
        demand_distributions = other.demand_distributions;
}  /* -----  end of method HostParameters::HostParameters  (copy constructor)  ----- */


/*
 *------------------------------------------------------------------------------
 *       Class:  HostParameters
 *      Method:  operator =
 * Description:  assignment operator
 *------------------------------------------------------------------------------
 */
HostParameters&
HostParameters::operator = ( const HostParameters &other ) {
    if ( this != &other ) {
        for (int i = 0; i < num_params_; ++i){
            params_[i] = other.params_[i];
        }
        demand_distributions = other.demand_distributions;
    }
    return *this;
}
/* :TODO:Wed Aug  5 23:44:20 2015:huangzonghao:
 *  bookmark
 */
/* reload to get the values from cuda device directly */
HostParameters&
HostParameters::operator = ( const DeviceParameters &device ) {
    if (device.IsComplete()){
        read_from_device(&T,       device.T,       1);
        read_from_device(&m,       device.m,       1);
        read_from_device(&k,       device.k,       1);
        read_from_device(&maxhold, device.maxhold, 1);
        read_from_device(&c,       device.c,       1);
        read_from_device(&h,       device.h,       1);
        read_from_device(&theta,   device.theta,   1);
        read_from_device(&r,       device.r,       1);
        read_from_device(&s,       device.s,       1);
        read_from_device(&alpha,   device.alpha,   1);
        read_from_device(&lambda,  device.lambda,  1);
    }
    return *this;
}  /* -----  end of method HostParameters::operator =  (assignment operator)  ----- */

/*
 *------------------------------------------------------------------------------
 *       Class:  HostParameters
 *      Method:  get_var_ptr
 * Description:  get the pointer to the internal variable.
 *------------------------------------------------------------------------------
 */
float * HostParameters::get_var_ptr (const char * var) {
    for (int i = 0; i < num_params_; ++i){
        if (var == param_names_[i]){
            return params_ + i;
        }
    }
    printf("Error: HostParameters doesn't have %s as a variable", var);
    return NULL;
}       /* -----  end of method HostParameters::get_var_ptr  ----- */

/*
 *------------------------------------------------------------------------------
 *       Class:  HostParameters
 *      Method:  get_distribution_ptr
 * Description:  returns the pointer to the specific distribution
 *------------------------------------------------------------------------------
 */
float * HostParameters::get_distribution_ptr (int index) {
    if(index + 1 > demand_distributions.size()){
        printf ("Error: the distribution index out of range !");
    }
    return demand_distributions[index].data();
}       /* -----  end of method HostParameters::get_distribution_ptr  ----- */

/*
 *------------------------------------------------------------------------------
 *       Class:  HostParameters
 *      Method:  set_param
 * Description:  An interface to set the value of the HostParameters
 *------------------------------------------------------------------------------
 */
bool HostParameters::set_param ( const char * var, float value ) {
    if (get_var_ptr(var) == NULL){
        printf("Error: cannot get the pointer of &s, set_param failed", var);
        return false;
    }
    *get_var_ptr(var) = value;
    return true;
}

bool HostParameters::set_param (int idx, float value) {
    if ( idx < params_.size() ){
        params_[idx] = value;
        return true;
    }
    else {
        printf("Error: the index for the parameter is out of range\n"
               "Failed to set_param by idx.\n");
        return false;
    }
}   /* -----  end of method HostParameters::set_param  ----- */


/*
 *------------------------------------------------------------------------------
 *       Class:  HostParameters
 *      Method:  set_distribution
 * Description:  set the distribution data
 *------------------------------------------------------------------------------
 */
bool HostParameters::set_distribution (int distriIdx, int valIdx, float val) {
    if (distriIdx + 1 > demand_distributions.size()){
        std::vector<float> temp;
        demand_distributions.push_back(temp);
    }
    if (valIdx == 0 && demand_distributions[distriIdx].size() != 0){
        demand_distributions[distriIdx].clear();
    }
    if (valIdx + 1 > demand_distributions[distriIdx].size()){
        demand_distributions[distriIdx].push_back(val);
    }
    else {
        demand_distributions[distriIdx][valIdx] = val;
    }
    return true;
}       /* -----  end of method HostParameters::set_distribution  ----- */


/*
 *------------------------------------------------------------------------------
 *       Class:  HostParameters
 *      Method:  get_value
 * Description:  return the variables' value stored in HostParameters
 *                 and note all the values are returned in float
 *------------------------------------------------------------------------------
 */
float HostParameters::get_value (const char * var) {
    if (get_var_ptr(var) == NULL){
        printf("Error: cannot get the pointer to %s, get_value failed.", var);
        return 0;
    }
    return *get_var_ptr(var);
}       /* -----  end of method HostParameters::get_value  ----- */


/*
 *------------------------------------------------------------------------------
 *       Class:  HostParameters
 *      Method:  print_params
 * Description:  print out the parameters stored in HostParameters
 *------------------------------------------------------------------------------
 */
void HostParameters::print_params () {
    printf("===========================\n");
    for (int i = 0; i < num_params_; ++i){
        printf("\e[1;33m%s : \e[38;5;166m%d\n", param_names_[i], params_[i]);
    }
    printf("\e[m===========================\n");
    printf("And the distributions : \n");
    for (int i = 0; i < demand_distributions.size(); ++i){
        printf("The No.%i distribution : \n", i);
        for (int j = 0; j < demand_distributions[i]; ++j){
            printf(" %d ", demand_distributions[i][j]);
        }
        printf("\n");
    }
    return ;
}       /* -----  end of method HostParameters::print_params  ----- */

/*
 *------------------------------------------------------------------------------
 *       Class:  HostParameters
 *      Method:  operator []
 * Description:  return the value of the parameters in float
 *                 And note this is both a accesor and a mutator
 *------------------------------------------------------------------------------
 */
float& HostParameters::operator [] (const char * var){
    return *get_var_ptr(var);
}       /* -----  end of method HostParameters::operator []  ----- */

/*
 *------------------------------------------------------------------------------
 *       Class:  HostParameters
 *      Method:  pop_param_names
 * Description:  pop up the parameters names for loading the paramters
 *------------------------------------------------------------------------------
 */
const char* HostParameters::pop_param_names (int idx) {
    if (idx < num_params_){
        return param_names_[i];
    }
    else return "NULL";
}       /* -----  end of method HostParameters::pop_param_names  ----- */

/* =============================================================================
 *                         end of file host_parameters.cc
 * =============================================================================
 */
