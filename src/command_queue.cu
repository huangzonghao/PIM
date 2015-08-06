/*
 * =============================================================================
 *
 *       Filename:  command_queue.cu
 *
 *    Description:  This file contains the implementation of CommandQueue
 *
 *        Created:  Fri Jul 24 13:52:37 2015
 *       Modified:  Fri Aug  7 00:49:26 2015
 *
 *         Author:  Huang Zonghao
 *          Email:  coding@huangzonghao.com
 *
 * =============================================================================
 */
#include <cuda.h>
#include <cuda_runtime.h>

#include "command_queue.h"
#include "../include/host_parameters.h"
#include "../include/device_parameters.h"

/*
 *------------------------------------------------------------------------------
 *       Class:  CommandQueue
 *      Method:  CommandQueue
 * Description:  constructor
 *------------------------------------------------------------------------------
 */
CommandQueue::CommandQueue () {
    *get_config_ptr("input_file_name")     = "params.json";
    *get_config_ptr("output_file_name")    = "output.txt";
    *get_config_ptr("output_format")       = "csv";
    *get_config_ptr("policy")              = "all";
    *get_config_ptr("recovery_file_name")  = "";
    *get_config_ptr("loggirg_file_name")   = "";
    *get_config_ptr("recording_file_name") = "";

    verbose_enabled_   = false;
    recovery_enabled_  = false;
    logging_enabled_   = false;
    recording_enabled_ = false;
    print_help_        = false;
    commands_loaded_   = false;
    parameters_loaded_ = false;

    host_params_   = new HostParameters;
    device_params_ = new DeviceParameters;

}  /* -----  end of method CommandQueue::CommandQueue  (constructor)  ----- */

/*
 *------------------------------------------------------------------------------
 *       Class:  CommandQueue
 *      Method:  CommandQueue
 * Description:  copy constructor
 *------------------------------------------------------------------------------
 */
CommandQueue::CommandQueue ( const CommandQueue &other ) {
    host_params_   = new HostParameters;
    device_params_ = new DeviceParameters;
    if (this != other){
        *host_params_ = *other.get_host_param_pointer();
        update_device_params();
        for (int i = 0; i < num_configs_; ++i){
            configs_[i] = other.configs_[i]
        }
        verbose_enabled_     = other.verbose_enabled_;
        recovery_enabled_    = other.recovery_enabled_;
        logging_enabled_     = other.logging_enabled_;
        recording_enabled_   = other.recording_enabled_;
        print_help_          = other.print_help_;
        commands_loaded_     = other.commands_loaded_;
        parameters_loaded_   = other.parameters_loaded_;
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
    if (this != other){
        *host_params_ = *other.get_host_param_pointer();
        update_device_params();
        for (int i = 0; i < num_configs_; ++i){
            configs_[i] = other.configs_[i]
        }
        verbose_enabled_     = other.verbose_enabled_;
        recovery_enabled_    = other.recovery_enabled_;
        logging_enabled_     = other.logging_enabled_;
        recording_enabled_   = other.recording_enabled_;
        print_help_          = other.print_help_;
        commands_loaded_     = other.commands_loaded_;
        parameters_loaded_   = other.parameters_loaded_;
    }
    return *this;
}  /* -----  end of method CommandQueue::operator =  (assignment operator)  ----- */

/*
 *------------------------------------------------------------------------------
 *       Class:  CommandQueue
 *      Method:  get_config_ptr
 * Description:  return the pointer to the configuration strings
 *------------------------------------------------------------------------------
 */
std::string * CommandQueue::get_config_ptr (const std::string &var) {
    for (int i = 0; i < num_configs_; ++i){
        if (var == config_names_[i])
            return configs + i;
    }
    printf("Error: %s is not a config variable name, return NULL.", var);
    return NULL;
}       /* -----  end of method CommandQueue::get_config_ptr  ----- */

/*
 *------------------------------------------------------------------------------
 *       Class:  CommandQueue
 *      Method:  get_config
 * Description:  return the configuration attributes
 *------------------------------------------------------------------------------
 */
std::string CommandQueue::get_config (const std::string &var) {
    if (get_config_ptr(var) == NULL){
        printf("Error: Cannot get the string of the %s.", var);
        return "";
    }
    return *get_config(var);
}       /* -----  end of method CommandQueue::get_config  ----- */

/*
 *------------------------------------------------------------------------------
 *       Class:  CommandQueue
 *      Method:  check_command
 * Description:  check whether we need to opearte certain commands
 *------------------------------------------------------------------------------
 */
bool CommandQueue::check_command (const std::string &var) {
    switch (var) {
        case "verbose":
            return verbose_enabled_;
        case "recovery":
            return recovery_enabled_;
        case "log":
            return logging_enabled_;
        case "record":
            return recording_enabled_;
        case "print_help":
            return print_help_;
        default:
            printf("Error: Invalide command name, check_command failed.\n");
            exit();
    }            /* -----  end switch  ----- */
}       /* -----  end of method CommandQueue::check_command  ----- */

/*
 *------------------------------------------------------------------------------
 *       Class:  CommandQueue
 *      Method:  check_status
 * Description:  check whether the necessary components has been loaded to
 *                 the class
 *------------------------------------------------------------------------------
 */
bool CommandQueue::check_status (const std::string &var) {
    switch(var){
        case "cmd":
            return commands_loaded_;
        case "param":
            return parameters_loaded_;
        default:
            printf("Error: %s is not a valid status variable, exit", var);
            exit();
    }
}       /* -----  end of method CommandQueue::check_status  ----- */

/*
 *------------------------------------------------------------------------------
 *       Class:  CommandQueue
 *      Method:  get_host_param_pointer
 * Description:  return the pointer to the HostParameters
 *------------------------------------------------------------------------------
 */
HostParameters * CommandQueue::get_host_param_pointer () {
    return host_params_;
}       /* -----  end of method CommandQueue::get_host_param_pointer  ----- */

/*
 *------------------------------------------------------------------------------
 *       Class:  CommandQueue
 *      Method:  get_host_param_value
 * Description:  return the value stored in HostParameters
 *------------------------------------------------------------------------------
 */
float CommandQueue::get_host_param_value (const std::string &var) {
    return *host_params_[var];
}       /* -----  end of method CommandQueue::get_host_param_value  ----- */


/*
 *------------------------------------------------------------------------------
 *       Class:  CommandQueue
 *      Method:  get_device_param_pointer
 * Description:  return the pointer to the DeviceParameters
 *------------------------------------------------------------------------------
 */
DeviceParameters * CommandQueue::get_device_param_pointer () {
    return device_params_;
}       /* -----  end of method CommandQueue::get_device_param_pointer  ----- */

/*
 *------------------------------------------------------------------------------
 *       Class:  CommandQueue
 *      Method:  load_host_params
 * Description:  load the specific value to HostParameters
 *------------------------------------------------------------------------------
 */
bool CommandQueue::load_host_params ( const std::string &var, float value ) {
    if (host_params_->set_value(var,value)) {
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
    *device_params_ = *host_params_;
    return true;
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
    *host_params_ = *device_params_;
    return true;
}       /* -----  end of method CommandQueue::retrieve_device_params  ----- */

/*
 *------------------------------------------------------------------------------
 *       Class:  CommandQueue
 *      Method:  load_commands
 * Description:  set the control parameters correspondingly
 *------------------------------------------------------------------------------
 */
bool CommandQueue::load_commands (const std::string var, const std::string value) {
    switch (var) {
        case "INPUT_FILE":
            *get_config_ptr("input_file_name") = value;
            break;
        case "OUPUT_FILE":
            *get_config_ptr("output_file_name") = value;
            break;
        case "OUPUT_FORMAT":
            *get_config_ptr("output_file_format") = value;
            break;
        case "POLICY":
            *get_config_ptr("policy") = value;
            break;
        case "RECOVERY":
            *get_config_ptr("recovery_file_name") = value;
            recovery_enabled_ = true;
            break;
        case "RECORDING":
            *get_config_ptr("recording_file_name") = value;
            recording_enabled_ = true;
            break;
        case "ENABLE_VERBOSE":
            verbose_enabled_ = true;
            break;
        case "LOGGING":
            *get_config_ptr("logging_file_name") = value;
            logging_enabled_ = true;
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


/* =============================================================================
 *                         end of file command_queue.cu
 * =============================================================================
 */
