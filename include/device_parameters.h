/*
 * =============================================================================
 *
 *       Filename:  device_parameters.h
 *
 *    Description:  The definition of DeviceParameters
 *
 *        Created:  Tue Jul 28 14:56:03 2015
 *       Modified:  Mon Aug 31 22:31:03 2015
 *
 *         Author:  Huang Zonghao
 *          Email:  coding@huangzonghao.com
 *
 * =============================================================================
 */
#ifndef DEVICE_PARAMETERS_H_
#define DEVICE_PARAMETERS_H_

#include <stdlib.h>
#include <stdio.h>
#include <cstring>
struct DemandDistribution;

struct DeviceParameters{
    size_t T;
    size_t m;
    size_t k;
    size_t maxhold;
    size_t num_distri;
    size_t table_length;
    float  c;
    float  h;
    float  theta;
    float  r;
    float  s;
    float  alpha;
    float  lambda;

    /* note this is not the value table, so of course this variable shall be able
     * to be dereferenced in the kernel!!!!
     * so this is a pointer pointing to some device memory
     */
    DemandDistribution **demand_distributions;
};

/*
 * ===  INL-FUNCTION  ==========================================================
 *         Name:  GetDeviceParameterValue
 *  Description:  return the value of the corresponding parameters stored in
 *                  DeviceParameters, the return type is float for convenience
 *       @param:  variable name
 *      @return:  float value
 * =============================================================================
 */
inline float GetDeviceParameterValue(DeviceParameters &d, const char *var){
    if(strcmp(var, "T") == 0){
        return d.T;
    }
    if(strcmp(var, "m") == 0){
        return d.m;
    }
    if(strcmp(var, "k") == 0){
        return d.k;
    }
    if(strcmp(var, "maxhold") == 0){
        return d.maxhold ;
    }
    if(strcmp(var, "num_distri") == 0){
        return d.num_distri;
    }
    if(strcmp(var, "table_length") == 0){
        return d.table_length;
    }
    if(strcmp(var,  "c") == 0){
        return d.c;
    }
    if(strcmp(var,  "h") == 0){
        return d.h;
    }
    if(strcmp(var,  "theta") == 0){
        return d.theta;
    }
    if(strcmp(var,  "r") == 0){
        return d.r;
    }
    if(strcmp(var,  "s") == 0){
        return d.s;
    }
    if(strcmp(var,  "alpha") == 0){
        return d.alpha;
    }
    if(strcmp(var,  "lambda") == 0){
        return d.lambda;
    }
    printf("Error: there is no variable named %s\n", var);
    return 0;
}       /* -----  end of inline function GetDeviceParameterValue  ----- */
/*
 * ===  FUNCTION  ==============================================================
 *         Name:  CopyDeviceParameters
 *  Description:  copy the structure of DeviceParameters
 *       @param:  source target
 *      @return:  bool
 * =============================================================================
 */
/* inline bool CopyDeviceParameters(DeviceParameters &source, DeviceParameters &target){
 *     target.T            = source.T;
 *     target.m            = source.m;
 *     target.k            = source.k;
 *     target.maxhold      = source.maxhold;
 *     target.num_distri   = source.num_distri;
 *     target.table_length = source.table_length;
 *     target.c            = source.c;
 *     target.h            = source.h;
 *     target.theta        = source.theta;
 *     target.r            = source.r;
 *     target.s            = source.s;
 *     target.alpha        = source.alpha;
 *     target.lambda       = source.lambda;
 *
 *     [> now start to copy the demand_distributions <]
 *     if (target.demand_distributions != NULL){
 *         delete target.demand_distributions;
 *         target.demand_distributions = new DemandDistribution[source.num_distri];
 *     }
 *     for(int i = 0; i < (int)source.num_distri; ++i){
 *         CopyDemandDistribution(source.demand_distributions[i], target.demand_distributions[i]);
 *     }
 *     return true;
 * }       [> -----  end of function CopyDeviceParameters  ----- <]
 */


#endif   /* ----- #ifndef DEVICE_PARAMETERS_H_  ----- */
/* =============================================================================
 *                         end of file device_parameters.h
 * =============================================================================
 */
