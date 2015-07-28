/*
 * =============================================================================
 *
 *       Filename:  command_queue.h
 *
 *    Description:  This file contains the classes holding all the configuration
 *                    parameters and controlling information
 *
 *        Created:  Thu Jul 23 00:45:56 2015
 *       Modified:  Tue Jul 28 16:50:24 2015
 *
 *         Author:  Huang Zonghao
 *          Email:  coding@huangzonghao.com
 *
 * =============================================================================
 */
#ifndef COMMAND_QUEUE_H_
#define COMMAND_QUEUE_H_
#include <stdlib.h>

#include "host_parameters.h"
#include "device_parameters.h"

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
    CommandQueue( const HostParameters &hp ) : h(hp){};
    /* copy constructor */
    CommandQueue ( const CommandQueue &other );
    /* destructor */
    ~CommandQueue ();

    /* =========================   ACCESSORS   =============================== */
    HostParameters * get_host_params();
    DeviceParameters * get_device_params();

    /* =========================   MUTATORS    =============================== */
    bool load_host_params(char * var, float value);

    /* =========================   OPERATORS   =============================== */
    bool update_device_params();
    bool retrieve_device_params();

    /* assignment operator */
    CommandQueue& operator = ( const CommandQueue &other );

  private:
    /* ========================  DATA MEMBERS  =============================== */

    HostParameters h;
    DeviceParameters d;

}; /* -----  end of class CommandQueue  ----- */




#endif   /* ----- #ifndef COMMAND_QUEUE_H_  ----- */


/* =============================================================================
 *                         end of file command_queue.h
 * =============================================================================
 */
