/*
 * =============================================================================
 *
 *       Filename:  command_queue.cu
 *
 *    Description:  This file contains the implementation of CommandQueue
 *
 *        Created:  Fri Jul 24 13:52:37 2015
 *       Modified:  Sat Aug  1 13:39:22 2015
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
 *      Method:  get_host_param_pointer
 * Description:  return the pointer to the HostParameters
 *------------------------------------------------------------------------------
 */
HostParameters * CommandQueue::get_host_param_pointer () {
    return &h;
}       /* -----  end of method CommandQueue::get_host_param_pointer  ----- */

/*
 *------------------------------------------------------------------------------
 *       Class:  CommandQueue
 *      Method:  get_host_param_value
 * Description:  return the value stored in HostParameters
 *------------------------------------------------------------------------------
 */
float CommandQueue::get_host_param_value (const char * var) {
    return h.get_value(var);
}       /* -----  end of method CommandQueue::get_host_param_value  ----- */


/*
 *------------------------------------------------------------------------------
 *       Class:  CommandQueue
 *      Method:  get_device_param_pointer
 * Description:  return th e pointer to the DeviceParameters
 *------------------------------------------------------------------------------
 */
DeviceParameters * CommandQueue::get_device_param_pointer () {
    return &d;
}       /* -----  end of method CommandQueue::get_device_param_pointer  ----- */

/*
 *------------------------------------------------------------------------------
 *       Class:  CommandQueue
 *      Method:  load_host_params
 * Description:  load the specific value to HostParameters
 *------------------------------------------------------------------------------
 */
bool CommandQueue::load_host_params ( const char * var, float value ) {
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
bool CommandQueue::retrieve_device_params () {
    h = b;
    return ;
}       /* -----  end of method CommandQueue::retrieve_device_params  ----- */

/*
 *------------------------------------------------------------------------------
 *       Class:  CommandQueue
 *      Method:  load_commands
 * Description:  set the control parameters correspondingly
 *------------------------------------------------------------------------------
 */
bool CommandQueue::load_commands (const char * var, const char * value) {
    switch (var) {
    case "INPUT_FILE":
        input_file_name_ = value;
        break;
    case "OUPUT_FILE":
        output_file_name_ = value;
        break;
    case "OUPUT_FORMAT":
        output_file_format_ = value;
            break;
    case "POLICY":
        policy_ = value;
        break;
    case "RECOVERY":
        recovery_file_ = value;
        is_recovery_enabled_ = true;
        break;
    case "RECORDING":
        recording_file_ = value;
        is_recording_enabled_ = true;
        break;
    case "ENABLE_VERBOSE":
        is_verbose_enabled_ = true;
        break;
    case "LOGGING":
        logging_file_ = value;
        is_logging_enabled_ = true;
        break;
    case "PRINT_HELP":
        print_help_ = true;
        break;
    default:
        printf("%s is not a valid attribute of CommandQueue, exit", var);
        return false;
        break;
}            /* -----  end switch  ----- */
    return true;
}       /* -----  end of method CommandQueue::load_commands  ----- */

/*
 *------------------------------------------------------------------------------
 *       Class:  CommandQueue
 *      Method:  do_print_help
 * Description:  return true if we need to print the help, otherwise false
 *------------------------------------------------------------------------------
 */
bool CommandQueue::do_print_help () {
    return print_help_;
}       /* -----  end of method CommandQueue::do_print_help  ----- */

/*
 *------------------------------------------------------------------------------
 *       Class:  CommandQueue
 *      Method:  do_verbose
 * Description:  return true if the verbose mode is enabled, otherwise false
 *------------------------------------------------------------------------------
 */
bool CommandQueue::do_verbose () {
    return is_verbose_enabled_;
}       /* -----  end of method CommandQueue::do_verbose  ----- */

/*
 *------------------------------------------------------------------------------
 *       Class:  CommandQueue
 *      Method:  get_input_file_name
 * Description:  return the input file stored in the CommandQueue
 *------------------------------------------------------------------------------
 */
const char * CommandQueue::get_input_file_name () {
    return input_file_name_.c_str();
}       /* -----  end of method CommandQueue::get_input_file_name  ----- */

/*
 *------------------------------------------------------------------------------
 *       Class:  CommandQueue
 *      Method:  get_output_file_name
 * Description:  return the output file name stored in the CommandQueue
 *------------------------------------------------------------------------------
 */
const char *  CommandQueue::get_output_file_name () {
    return output_file_name_.c_str();
}       /* -----  end of method CommandQueue::get_output_file_name  ----- */

/*
 *------------------------------------------------------------------------------
 *       Class:  CommandQueue
 *      Method:  get_output_format
 * Description:  return the output format stored in the CommandQueue
 *------------------------------------------------------------------------------
 */
const char * CommandQueue::get_output_format () {
    return output_file_format_.c_str();
}       /* -----  end of method CommandQueue::get_output_format  ----- */

/*
 *------------------------------------------------------------------------------
 *       Class:  CommandQueue
 *      Method:  get_policy
 * Description:  return the policy stored in the CommandQueue
 *------------------------------------------------------------------------------
 */
const char * CommandQueue::get_policy () {
    return policy_.c_str();
}       /* -----  end of method CommandQueue::get_policy  ----- */

/*
 *------------------------------------------------------------------------------
 *       Class:  CommandQueue
 *      Method:  get_recovery_file_name
 * Description:  return the recovery file name stored in the CommandQueue
 *------------------------------------------------------------------------------
 */
const char * CommandQueue::get_recovery_file_name () {
    return recovery_file_.c_str();
}       /* -----  end of method CommandQueue::get_recovery_file_name  ----- */

/*
 *------------------------------------------------------------------------------
 *       Class:  CommandQueue
 *      Method:  get_logging_file_name
 * Description:  get the name of logging file stored in the CommandQueue
 *------------------------------------------------------------------------------
 */
const char * CommandQueue::get_logging_file_name () {
    return logging_file_.c_str();
}       /* -----  end of method CommandQueue::get_logging_file_name  ----- */

/*
 *------------------------------------------------------------------------------
 *       Class:  CommandQueue
 *      Method:  get_recording_file_name
 * Description:  get the name of recording file stored in the CommandQueue
 *------------------------------------------------------------------------------
 */
const char * CommandQueue::get_recording_file_name () {
    return recording_file_.c_str();
}       /* -----  end of method CommandQueue::get_recording_file_name  ----- */



/* =============================================================================
 *                         end of file command_queue.cu
 * =============================================================================
 */
