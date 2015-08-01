/*
 * =============================================================================
 *
 *       Filename:  host_parameters.h
 *
 *    Description:  The definition of HostParameters
 *
 *        Created:  Tue Jul 28 14:54:25 2015
 *       Modified:  Sat Aug  1 13:58:21 2015
 *
 *         Author:  Huang Zonghao
 *          Email:  coding@huangzonghao.com
 *
 * =============================================================================
 */
#ifndef HOST_PARAMETERS_H_
#define HOST_PARAMETERS_H_
#include <stdlib.h>
#include "device_parameters.h"
/*
 * =============================================================================
 *        Class:  HostParameters
 *  Description:  This class contains the host copy of the configuration params
 * =============================================================================
 */
class HostParameters
{
  public:
/* :TODO:Sat Aug  1 13:58:00 2015:huangzonghao:
 *  hey, i think i really should use the std container in here
 */
    /* constructor */
    HostParameters ();
    /* copy constructor */
    HostParameters ( const HostParameters &other );
    /* destructor */
    ~HostParameters ();

    /* =========================   ACCESSORS   =============================== */
    bool set_value(const char*var, size_t value);
    bool set_value(const char*var, float value);
    float get_value(const char * var);
    /* =========================   MUTATORS    =============================== */

    /* =========================   OPERATORS   =============================== */

    /* assignment operator */
    HostParameters& operator = ( const HostParameters &other );
    HostParameters& operator = ( const DeviceParameters &other );

  protected:
    /* ========================  DATA MEMBERS  =============================== */

  private:
    /* ========================  DATA MEMBERS  =============================== */
    /* number of periods */
    size_t T;
    /* total number of categories */
    size_t m;
    /* maximum number for each category */
    size_t k;
    /* maximum storage */
    size_t maxhold;
    /* the ordering cost of each item */
    float c;
    /* storing cost for each item */
    float h;
    /* the disposal cost of each item */
    float theta;
    /* the price of each item */
    float r;
    /* the salvage benefit for one item */
    float s;
    /* the discount rate */
    float alpha ;
    /* the arrival rate for poisson distribution */
    float lambda;

    size_t max_demand;
    size_t min_demand;
    float * demand_distribution;

}; /* -----  end of class HostParameters  ----- */


#endif   /* ----- #ifndef HOST_PARAMETERS_H_  ----- */

/* =============================================================================
 *                         end of file host_parameters.h
 * =============================================================================
 */
