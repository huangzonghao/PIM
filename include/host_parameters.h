/*
 * =============================================================================
 *
 *       Filename:  host_parameters.h
 *
 *    Description:  The definition of HostParameters
 *
 *        Created:  Tue Jul 28 14:54:25 2015
 *       Modified:  Mon Aug 31 00:32:48 2015
 *
 *         Author:  Huang Zonghao
 *          Email:  coding@huangzonghao.com
 *
 * =============================================================================
 */
#ifndef HOST_PARAMETERS_H_
#define HOST_PARAMETERS_H_
#include <stdlib.h>
#include <string>
#include <vector>
#include "../include/demand_distribution.h"

struct DeviceParameters;

/*
 * =============================================================================
 *        Class:  HostParameters
 *  Description:  This class contains the host copy of the configuration params
 * =============================================================================
 */
class HostParameters
{
  public:
/* :REMARKS:Wed Aug  5 22:52:35 2015:huangzonghao:
 *  store all the parameters as float and only convert them to float
 *  when passing them to the device
 */

    /* constructor */
    HostParameters ();
    /* copy constructor */
    HostParameters ( const HostParameters &other );
    /* destructor */
    ~HostParameters ();

    /* =========================   ACCESSORS   =============================== */
    float get_value(const char *var);
    float& operator [](const char *var); /* this is also mutator */
    DemandDistribution *get_distribution_ptr(int index);
    const char *pop_param_names(int idx); /* will return NULL if out of range */
    const int &get_param_num(){return num_params_;};
    /* float *get_distribution_ptr(int index); */

    /* =========================   MUTATORS    =============================== */
    bool set_param(const char *var, float value);
    /* to set the parameters in batch mode */
    bool set_param(int idx, float value);
    /* the special arrangement for loading the demand_distribution */
    bool set_distribution (int distriIdx,
                           float *val,
                           size_t min_demand,
                           size_t max_demand);
    /* bool load_distributation(size_t id, float * array, size_t length); */
    /* =========================   OPERATORS   =============================== */

    /* assignment operator */
    HostParameters& operator = ( const HostParameters &other );
    HostParameters& operator = ( const DeviceParameters &other );
    void print_params();

  protected:
    /* ========================  DATA MEMBERS  =============================== */

  private:
    /* ========================  DATA MEMBERS  =============================== */

   /*  Paramters Lists
    *  Notice : all the variables are stored as float in the class, and will
    *            converted to size_t while passing to device if necessary
    *
    *   No.   Variable Name       Description
    *    0    size_t T            number of periods
    *    1    size_t m            total number of categories
    *    2    size_t k            maximum number for each category
    *    3    size_t maxhold      maximum storage
    *    4    size_t num_distri   the total number of distributions
    *    5    float  c            the ordering cost of each item
    *    6    float  h            storing cost for each item
    *    7    float  theta        the disposal cost of each item
    *    8    float  r            the price of each item
    *    9    float  s            the salvage benefit for one item
    *   10    float  alpha        the discount rate
    *   11    float  lambda       the arrival rate for poisson distribution
    *
    */

    const int num_params_ = 12;

    /* the following is a c++11 feature, so the compiler support is required */
    const char *param_names_[12] = { "T",
                                     "m",
                                     "k",
                                     "maxhold",
                                     "num_distri",
                                     "c",
                                     "h",
                                     "theta",
                                     "r",
                                     "s",
                                     "alpha",
                                     "lambda"};
    float params_[12];
    /* std::vector< std::vector<float> > demand_distributions; */
    std::vector<DemandDistribution> demand_distributions_;

    float *get_var_ptr(const char *var);
}; /* -----  end of class HostParameters  ----- */


#endif   /* ----- #ifndef HOST_PARAMETERS_H_  ----- */

/* =============================================================================
 *                         end of file host_parameters.h
 * =============================================================================
 */
