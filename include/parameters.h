/*
 * =============================================================================
 *
 *       Filename:  parameters.h
 *
 *    Description:  The declaration of the global variables which shall be read
 *                    from the json file
 *
 *        Created:  Thu Jul 23 00:24:41 2015
 *       Modified:  Thu Jul 23 00:24:41 2015
 *
 *         Author:  Huang Zonghao
 *          Email:  coding@huangzonghao.com
 *
 * =============================================================================
 */
#ifndef PARAMETERS_H
#define PARAMETERS_H
/* the parameters of the program */
/* total number of categories */
extern size_t h_m;
extern size_t * d_m;

/* maximum number for each category */
extern size_t h_k;
extern size_t * d_k;

/* number of periods */
extern size_t h_T;
extern size_t * d_T;

/* storing cost for each item */
extern float h_h;
extern float * d_h;

/* the price of each item */
extern float h_r;
extern float * d_r;

/* the ordering cost of each item */
extern float h_c;
extern float * d_c;

/* the disposal cost of each item */
extern float h_theta;
extern float * d_theta;

/* the salvage benefit for one item */
extern float h_s;
extern float * d_s;

/* the discount rate */
extern float h_alpha;
extern float * d_alpha;

/* maximum storage */
extern size_t h_maxhold;
extern size_t * d_maxhold;

/* the arrival rate for Poisson distribution */
extern float h_lambda;
extern float * d_lambda;

extern size_t h_min_demand;
extern size_t * d_min_demand;

extern size_t h_max_demand;
extern size_t * d_max_demand;

extern float *h_demand_distribution;


#endif
