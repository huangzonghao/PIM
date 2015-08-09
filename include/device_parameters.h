/*
 * =============================================================================
 *
 *       Filename:  device_parameters.h
 *
 *    Description:  The definition of DeviceParameters
 *
 *        Created:  Tue Jul 28 14:56:03 2015
 *       Modified:  Sun Aug  9 03:49:32 2015
 *
 *         Author:  Huang Zonghao
 *          Email:  coding@huangzonghao.com
 *
 * =============================================================================
 */
#ifndef DEVICE_PARAMETERS_H_
#define DEVICE_PARAMETERS_H_

#include <stdlib.h>

struct DeviceParameters{
    size_t T;
    size_t m;
    size_t k;
    size_t maxhold;
    size_t max_demand;
    size_t min_demand;
    size_t num_distri;
    size_t table_length;
    float  c;
    float  h;
    float  theta;
    float  r;
    float  s;
    float  alpha;
    float  lambda;

    float ** demand_distributions;
};
#endif   /* ----- #ifndef DEVICE_PARAMETERS_H_  ----- */
/* =============================================================================
 *                         end of file device_parameters.h
 * =============================================================================
 */
