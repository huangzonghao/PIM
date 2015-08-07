/*
 * =============================================================================
 *
 *       Filename:  device_parameters.h
 *
 *    Description:  The definition of DeviceParameters
 *
 *        Created:  Tue Jul 28 14:56:03 2015
 *       Modified:  Fri Aug  7 13:14:30 2015
 *
 *         Author:  Huang Zonghao
 *          Email:  coding@huangzonghao.com
 *
 * =============================================================================
 */
#ifndef DEVICE_PARAMETERS_H_
#define DEVICE_PARAMETERS_H_

#include <stdlib.h>
#include <string>
#include <vector>

class HostParameters;

/*
 * =============================================================================
 *        Class:  DeviceParameters
 *  Description:  This class contains the pointer to the device configure params
 * =============================================================================
 */
class DeviceParameters
{
  public:
    /* =========================   LIFECYCLE   =============================== */

    /* constructor */
    DeviceParameters ();
    DeviceParameters ( HostParameters * );
    /* copy constructor */
    DeviceParameters ( const DeviceParameters &other );
    /* destructor */
    ~DeviceParameters ();
    /* valid the completness of the structure */

    /* =========================   ACCESSORS   =============================== */
    float get_value(const char * var);
    float operator [] ( const char * var );
    float *  get_float_ptr(const char * var);
    float ** get_var_ptr(const char * var);

    bool is_complete();
    bool is_owner();
    bool is_linked();
    /* =========================   MUTATORS    =============================== */
    bool set_value( const char * var, float value );

    /* =========================   OPERATORS   =============================== */

    /* assignment operator */
    DeviceParameters& operator = ( const DeviceParameters &other );
    DeviceParameters& operator = ( const HostParameters &other );

  protected:
    /* ========================  DATA MEMBERS  =============================== */

  private:
    /* this denotes whether the object has pointed to a device copy */
    bool is_target_set_ = false;
    /*
     * this denotes whether the object is the owner of the device copy
     * only the owner will free the device memory when being destructed
     * the owner ship can only be obtained at construction or being assigned
     * by the = operator
     */
    bool is_owner_ = false;

    /* ========================  DATA MEMBERS  =============================== */

   /*  Paramters Lists
    *  Notice : all the variables are stored as float in the class, and will
    *            converted to size_t while passing to device if necessary
    *
    *   No.   Variable Name        Description
    *    0    size_t * T           number of periods
    *    1    size_t * m           total number of categories
    *    2    size_t * k           maximum number for each category
    *    3    size_t * maxhold     maximum storage
    *    4    size_t * max_demand  the maximum possible demands of a day
    *    5    size_t * min_demand  the minimum possible demands of a day
    *    6    size_t * num_distri  the total number of distributions
    *    7    float  * c           the ordering cost of each item
    *    8    float  * h           storing cost for each item
    *    9    float  * theta       the disposal cost of each item
    *   10    float  * r           the price of each item
    *   11    float  * s           the salvage benefit for one item
    *   12    float  * alpha       the discount rate
    *   13    float  * lambda      the arrival rate for poisson distribution
    *
    */
    const int num_params_ = 14;
    const char *param_names_[14] = { "T",
                                     "m",
                                     "k",
                                     "maxhold",
                                     "max_demand",
                                     "min_demand",
                                     "num_distri",
                                     "c",
                                     "h",
                                     "theta",
                                     "r",
                                     "s",
                                     "alpha",
                                     "lambda"};
    float ** params_[14];
    std::vector<float *> demand_distributions;

}; /* -----  end of class DeviceParameters  ----- */



#endif   /* ----- #ifndef DEVICE_PARAMETERS_H_  ----- */
/* =============================================================================
 *                         end of file device_parameters.h
 * =============================================================================
 */
