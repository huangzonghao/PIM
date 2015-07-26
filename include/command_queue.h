/*
 * =============================================================================
 *
 *       Filename:  command_queue.h
 *
 *    Description:  This file contains the classes holding all the configuration
 *                    parameters and controlling information
 *
 *        Created:  Thu Jul 23 00:45:56 2015
 *       Modified:  Mon Jul 27 04:24:16 2015
 *
 *         Author:  Huang Zonghao
 *          Email:  coding@huangzonghao.com
 *
 * =============================================================================
 */
#ifndef COMMAND_QUEUE_H_
#define COMMAND_QUEUE_H_
/* :TODO:Mon Jul 27 03:17:32 2015:huangzonghao:
 * the parameterloading should be done by the command queue
 */
/*
 * =============================================================================
 *        Class:  CommandQueue
 *  Description:  This class contains all the configuration and control
 *                  information of the task
 * =============================================================================
 */
class CommandQueue
{
  public:
    /* =========================   LIFECYCLE   =============================== */

    /* constructor */
    CommandQueue ();
    /* copy constructor */
    CommandQueue ( const CommandQueue &other );
    /* destructor */
    ~CommandQueue ();

    /* =========================   ACCESSORS   =============================== */

    /* =========================   MUTATORS    =============================== */

    /* =========================   OPERATORS   =============================== */

    /* assignment operator */
    CommandQueue& operator = ( const CommandQueue &other );

  private:
    /* ========================  DATA MEMBERS  =============================== */

    HostParameters h;
    DeviceParameters d;

}; /* -----  end of class CommandQueue  ----- */


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
    /* =========================   LIFECYCLE   =============================== */

    /* constructor */
    HostParameters ();
    /* copy constructor */
    HostParameters ( const HostParameters &other );
    /* destructor */
    ~HostParameters ();

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

    /* =========================   OPERATORS   =============================== */

    /* assignment operator */
    HostParameters& operator = ( const HostParameters &other );
    HostParameters& operator = ( const DeviceParameters &other );

}; /* -----  end of class HostParameters  ----- */


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
    bool IsComplete();

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
    /* =========================   OPERATORS   =============================== */
    
    /* assignment operator */
    DeviceParameters& operator = ( const DeviceParameters &other );
    DeviceParameters& operator = ( const HostParameters &other );
    bool set_value( char* var, float value );
  private:
    /* this denotes whether the object has pointed to a device copy */
    int is_target_set = 0;
    /*
     * this denotes whether the object is the owner of the device copy
     * only the owner will free the device memory when being destructed
     * the owner ship can only be obtained at construction or being assigned
     * by the = operator
     */
    int is_owner = 0;

}; /* -----  end of class DeviceParameters  ----- */




#endif   /* ----- #ifndef COMMAND_QUEUE_H_  ----- */


/* =============================================================================
 *                         end of file command_queue.h
 * =============================================================================
 */
