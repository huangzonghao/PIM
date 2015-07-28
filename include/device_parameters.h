/*
 * =============================================================================
 *
 *       Filename:  device_parameters.h
 *
 *    Description:  The definition of DeviceParameters
 *
 *        Created:  Tue Jul 28 14:56:03 2015
 *       Modified:  Tue Jul 28 17:44:33 2015
 *
 *         Author:  Huang Zonghao
 *          Email:  coding@huangzonghao.com
 *
 * =============================================================================
 */
#ifndef DEVICE_PARAMETERS_H_
#define DEVICE_PARAMETERS_H_

#include <stdlib.h>
#include "host_parameters.h"
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

    /* =========================   MUTATORS    =============================== */
    bool set_value( char* var, size_t value );
    bool set_value( char* var, float value );

    /* =========================   OPERATORS   =============================== */

    /* assignment operator */
    DeviceParameters& operator = ( const DeviceParameters &other );
    DeviceParameters& operator = ( const HostParameters &other );

    bool is_complete();
    bool is_owner();
    bool is_linked();

  protected:
    /* ========================  DATA MEMBERS  =============================== */

  private:
    /* this denotes whether the object has pointed to a device copy */
    int is_target_set_ = 0;
    /*
     * this denotes whether the object is the owner of the device copy
     * only the owner will free the device memory when being destructed
     * the owner ship can only be obtained at construction or being assigned
     * by the = operator
     */
    int is_owner_ = 0;

    /* ========================  DATA MEMBERS  =============================== */
    /* number of periods */
    size_t * T;
    /* total number of categories */
    size_t * m;
    /* maximum number for each category */
    size_t * k;
    /* maximum storage */
    size_t * maxhold;
    /* the ordering cost of each item */
    float * c;
    /* storing cost for each item */
    float * h;
    /* the disposal cost of each item */
    float * theta;
    /* the price of each item */
    float * r;
    /* the salvage benefit for one item */
    float * s;
    /* the discount rate */
    float * alpha ;
    /* the arrival rate for poisson distribution */
    float * lambda;

    size_t * max_demand;
    size_t * min_demand;
    float * demand_distribution;

}; /* -----  end of class DeviceParameters  ----- */



#endif   /* ----- #ifndef DEVICE_PARAMETERS_H_  ----- */
/* =============================================================================
 *                         end of file device_parameters.h
 * =============================================================================
 */
