/*
 * =============================================================================
 *
 *       Filename:  device_parameters.h
 *
 *    Description:  The definition of DeviceParameters
 *
 *        Created:  Tue Jul 28 14:56:03 2015
 *       Modified:  Wed Aug 26 13:48:32 2015
 *
 *         Author:  Huang Zonghao
 *          Email:  coding@huangzonghao.com
 *
 * =============================================================================
 */
#ifndef DEVICE_PARAMETERS_H_
#define DEVICE_PARAMETERS_H_

#include <stdlib.h>
#include "demand_distribution.h"

/* :TODO:Wed Aug 26 12:25:43 2015:huangzonghao:
 *  added a structure for DemandDistribution, so need to change the CommandQueue
 *  and other corresponding functions
 */
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
     */
    DemandDistribution *demand_distributions;
};

/*
 * ===  FUNCTION  ==============================================================
 *         Name:  CopyDeviceParameters
 *  Description:  copy the structure of DeviceParameters
 *       @param:  source target
 *      @return:  bool
 * =============================================================================
 */
inline bool CopyDeviceParameters(DeviceParameters &source, DeviceParameters &target){
    target.T            = source.T;
    target.m            = source.m;
    target.k            = source.k;
    target.maxhold      = source.maxhold;
    target.num_distri   = source.num_distri;
    target.table_length = source.table_length;
    target.c            = source.c;
    target.h            = source.h;
    target.theta        = source.theta;
    target.r            = source.r;
    target.s            = source.s;
    target.alpha        = source.alpha;
    target.lambda       = source.lambda;

    /* now start to copy the demand_distributions */
    if (target.demand_distributions != NULL){
        delete target.demand_distributions;
        target.demand_distributions = new DemandDistribution[source.num_distri];
    }
    for(int i = 0; i < (int)source.num_distri; ++i){
        CopyDemandDistribution(source.demand_distributions[i], target.demand_distributions[i]);
    }
    return true;
}       /* -----  end of function CopyDeviceParameters  ----- */


#endif   /* ----- #ifndef DEVICE_PARAMETERS_H_  ----- */
/* =============================================================================
 *                         end of file device_parameters.h
 * =============================================================================
 */
