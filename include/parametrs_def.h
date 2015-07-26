/*
 * =============================================================================
 *
 *       Filename:  parametrs_def.h
 *
 *    Description:  The defination of the global variables
 *
 *        Created:  Thu Jul 23 00:27:52 2015
 *       Modified:  Thu Jul 23 00:27:52 2015
 *
 *         Author:  Huang Zonghao
 *          Email:  coding@huangzonghao.com
 *
 * =============================================================================
 */
#ifndef PARAMETRS_DEF_H_
#define PARAMETRS_DEF_H_

/* total number of categories */
size_t h_m;
size_t * d_m;

/* maximum number for each category */
size_t h_k;
size_t * d_k;

/* number of periods */
size_t h_T;
size_t * d_T;

/* storing cost for each item */
float h_h;
float * d_h;

/* the price of each item */
float h_r;
float * d_r;

/* the ordering cost of each item */
float h_c;
float * d_c;

/* the disposal cost of each item */
float h_theta;
float * d_theta;

/* the salvage benefit for one item */
float h_s;
float * d_s;

/* the discount rate */
float h_alpha;
float * d_alpha;

/* maximum storage */
size_t h_maxhold;
size_t * d_maxhold;

/* the arrival rate for Poisson distribution */
float h_lambda;
float * d_lambda;

size_t h_min_demand;
size_t * d_min_demand;

size_t h_max_demand;
size_t * d_max_demand;

float *h_demand_distribution;


#endif   /* ----- #ifndef PARAMETRS_DEF_H_  ----- */

/* =============================================================================
 *                         end of file parametrs_def.h
 * =============================================================================
 */
