/*
 * =============================================================================
 *
 *       Filename:  host_parameters.h
 *
 *    Description:  The definition of HostParameters
 *
 *        Created:  Tue Jul 28 14:54:25 2015
 *       Modified:  Tue Jul 28 20:46:48 2015
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
 /* :TODO:Fri Jul 24 02:02:25 2015 02:02:huangzonghao:
  * do the copy part
  */
    /* constructor */
    HostParameters ();
    /* copy constructor */
    HostParameters ( const HostParameters &other );
    /* destructor */
    ~HostParameters ();

    /* =========================   ACCESSORS   =============================== */
    bool set_value(char*var, size_t value);
    bool set_value(char*var, float value);
    size_t get_int(char* var);
    float get_float(char* var);
    void * get_ptr(char* var);

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
