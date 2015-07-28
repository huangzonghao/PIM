/*
 * =============================================================================
 *
 *       Filename:  command_queue.cu
 *
 *    Description:  This file contains the implementation of CommandQueue
 *
 *        Created:  Fri Jul 24 13:52:37 2015
 *       Modified:  Tue Jul 28 20:46:42 2015
 *
 *         Author:  Huang Zonghao
 *          Email:  coding@huangzonghao.com
 *
 * =============================================================================
 */
#include <cuda.h>
#include <cuda_runtime.h>

#include "command_queue.h"

/*
 *------------------------------------------------------------------------------
 *       Class:  CommandQueue
 *      Method:  CommandQueue
 * Description:  copy constructor
 *------------------------------------------------------------------------------
 */
CommandQueue::CommandQueue ( const CommandQueue &other ) {
    if (this != other){
        h = other.h;
        if (other.d.IsComplete()){
            d = other.d;
        }
    }
}  /* -----  end of method CommandQueue::CommandQueue  (copy constructor)  ----- */


/*
 *------------------------------------------------------------------------------
 *       Class:  CommandQueue
 *      Method:  operator =
 * Description:  assignment operator
 *------------------------------------------------------------------------------
 */
CommandQueue&
CommandQueue::operator = ( const CommandQueue &other ) {
    if ( this != &other ) {
        h = other.h;
        if (other.d.IsComplete()){
            d = other.d;
        }
    }
    return *this;
}  /* -----  end of method CommandQueue::operator =  (assignment operator)  ----- */

/*
 *------------------------------------------------------------------------------
 *       Class:  CommandQueue
 *      Method:  get_host_params
 * Description:  return the pointer to the HostParameters
 *------------------------------------------------------------------------------
 */
HostParameters CommandQueue::get_host_params () {
    return &h;
}       /* -----  end of method CommandQueue::get_host_params  ----- */

/*
 *------------------------------------------------------------------------------
 *       Class:  CommandQueue
 *      Method:  get_device_params
 * Description:  return th e pointer to the DeviceParameters
 *------------------------------------------------------------------------------
 */
DeviceParameters CommandQueue::get_device_params () {
    return &d;
}       /* -----  end of method CommandQueue::get_device_params  ----- */

/*
 *------------------------------------------------------------------------------
 *       Class:  CommandQueue
 *      Method:  load_host_params
 * Description:  load the specific value to HostParameters
 *------------------------------------------------------------------------------
 */
bool CommandQueue::load_host_params ( char *var, float value ) {
    if (h.set_value(var,value)) {
        return true;
    }
    else return false;
}       /* -----  end of method CommandQueue::load_host_params  ----- */

/*
 *------------------------------------------------------------------------------
 *       Class:  CommandQueue
 *      Method:  update_device_params
 * Description:  passt the params stored in the HostParameters to
 *                 DeviceParameters
 *------------------------------------------------------------------------------
 */
bool CommandQueue::update_device_params () {
    d = h;
    return ;
}       /* -----  end of method CommandQueue::update_device_params  ----- */

/*
 *------------------------------------------------------------------------------
 *       Class:  CommandQueue
 *      Method:  retrieve_device_params
 * Description:  copy the params stored in the DeviceParameters back to
 *                 HostParameters
 *------------------------------------------------------------------------------
 */
<+FUNC_TYPE+> CommandQueue::retrieve_device_params ( <+argument list+> ) {
    <+FUNCTION+>
    return ;
}       /* -----  end of method CommandQueue::retrieve_device_params  ----- */











/* =============================================================================
 *                         end of file command_queue.cu
 * =============================================================================
 */
