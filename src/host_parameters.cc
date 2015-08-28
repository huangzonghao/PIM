/*
 * =============================================================================
 *
 *       Filename:  host_parameters.cc
 *
 *    Description:  The implementation of HostParameters
 *
 *        Created:  Tue Jul 28 14:58:27 2015
 *       Modified:  Fri Aug 28 05:48:57 2015
 *
 *         Author:  Huang Zonghao
 *          Email:  coding@huangzonghao.com
 *
 * =============================================================================
 */

#include "../include/host_parameters.h"
#include "../include/device_parameters.h"
#include "../include/demand_distribution.h"

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
        demand_distributions_ = other.demand_distributions_;
    }
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
        demand_distributions_ = other.demand_distributions_;
    }
    return *this;
}

/* reload to get the values from cuda device directly */
/* HostParameters&
 * HostParameters::operator = ( const DeviceParameters &device ) {
 *     *get_var_ptr("T") = device.T;
 *     *get_var_ptr("m") = device.m;
 *     *get_var_ptr("k") = device.k;
 *     *get_var_ptr("maxhold") = device.maxhold;
 *     *get_var_ptr("num_distri") = device.num_distri;
 *     *get_var_ptr("c") = device.c;
 *     *get_var_ptr("h") = device.h;
 *     *get_var_ptr("theta") = device.theta;
 *     *get_var_ptr("r") = device.r;
 *     *get_var_ptr("s") = device.s;
 *     *get_var_ptr("alpha") = device.alpha;
 *     *get_var_ptr("lambda") = device.lambda;
 *     return *this;
 * }  [> -----  end of method HostParameters::operator =  (assignment operator)  ----- <]
 */

/*
 *------------------------------------------------------------------------------
 *       Class:  HostParameters
 *      Method:  get_var_ptr
 * Description:  get the pointer to the internal variable.
 *------------------------------------------------------------------------------
 */
float *HostParameters::get_var_ptr (const char *var) {
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
DemandDistribution *HostParameters::get_distribution_ptr(int index){
    if(index + 1 > (int)demand_distributions_.size()){
        printf ("Error: the distribution index out of range !");
    }
    return demand_distributions_.data() + index;
}       /* -----  end of method HostParameters::get_distribution_ptr  ----- */

/*
 *------------------------------------------------------------------------------
 *       Class:  HostParameters
 *      Method:  set_param
 * Description:  An interface to set the value of the HostParameters
 *------------------------------------------------------------------------------
 */
/* set the variable by name */
bool HostParameters::set_param ( const char *var, float value ) {
    if (get_var_ptr(var) == NULL){
        printf("Error: cannot get the pointer of %s, set_param failed", var);
        return false;
    }
    *get_var_ptr(var) = value;
    return true;
}
/* set the varibale by index, convenient to be used in a for loop */
bool HostParameters::set_param (int idx, float value) {
    if ( idx < num_params_ ){
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
 *                 will set the one entire demand array at once
 *------------------------------------------------------------------------------
 */
bool HostParameters::set_distribution (int distriIdx,
                                       float *val,
                                       size_t min_demand,
                                       size_t max_demand){
    /* first check if the distriIdx is beyond the size of demand_distributions,
     *     if true then allocate a new space
     */
    if (distriIdx + 1 > (int)demand_distributions_.size()){
        DemandDistribution temp;
        demand_distributions_.push_back(temp);
    }
    /* now pass the entire array to the DemandDistribution
     *     and note the max length of the demand array is fixed to MAX_DISTRIBUTION_LENGTH
     *     check first
     */
    if(max_demand - min_demand > MAX_DISTRIBUTION_LENGTH){
        printf("Error: the distribution array is too long, please check the"
               "distribution array or modify the MAX_DISTRIBUTION_LENGTH macro"
               "in include/demand_distribution.h if necessary\n");
        return false;
    }
    for(size_t i = 0; i < max_demand - min_demand; ++i){
        demand_distributions_[distriIdx].table[i] = val[i];
    }
    /* lastly set the min_demand and the max_demand */
    demand_distributions_[distriIdx].min_demand = min_demand;
    demand_distributions_[distriIdx].max_demand = max_demand;
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
float HostParameters::get_value (const char *var) {
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
        printf("\e[1;33m%s : \e[38;5;166m%f\n", param_names_[i], params_[i]);
    }
    printf("\e[m===========================\n");
    printf("And the distributions : \n");
    for (int i = 0; i < (int)demand_distributions_.size(); ++i){
        printf("The No.%i distribution : \n", i);
        for (int j = 0;
             j < (int)(demand_distributions_[i].max_demand - demand_distributions_[i].min_demand);
             ++j){

            printf(" %f ", demand_distributions_[i].table[j]);
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
float& HostParameters::operator [] (const char *var){
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
        return param_names_[idx];
    }
    else return "NULL";
}       /* -----  end of method HostParameters::pop_param_names  ----- */

/* =============================================================================
 *                         end of file host_parameters.cc
 * =============================================================================
 */
