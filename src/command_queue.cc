/*
 * =============================================================================
 *
 *       Filename:  command_queue.cc
 *
 *    Description:  This file contains the implementation of CommandQueue
 *
 *        Created:  Fri Jul 24 13:52:37 2015
 *       Modified:  Sun Aug  9 10:22:18 2015
 *
 *         Author:  Huang Zonghao
 *          Email:  coding@huangzonghao.com
 *
 * =============================================================================
 */
#include <fstream>

#include <stdlib>
#include "../thirdparty/rapidjson/document.h"
#include "../thirdparty/rapidjson/prettywriter.h"
#include "../thirdparty/rapidjson/filereadstream.h"
#include "../thirdparty/rapidjson/filewritestream.h"

#include "../include/command_queue.h"
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
CommandQueue::CommandQueue ( CommandQueue &other ) {
    host_params_   = new HostParameters;
    device_params_ = new DeviceParameters;
    if (this != &other){
        *host_params_ = *(other.get_host_param_pointer());
        update_device_params();
        for (int i = 0; i < num_configs_; ++i){
            configs_[i] = other.configs_[i]
        }
        verbose_enabled_     = other.verbose_enabled_;
        recovery_enabled_    = other.recovery_enabled_;
        logging_enabled_     = other.logging_enabled_;
        recording_enabled_   = other.recording_enabled_;
        print_help_          = other.print_help_;
    }
}  /* -----  end of method CommandQueue::CommandQueue  (copy constructor)  ----- */

/*
 *------------------------------------------------------------------------------
 *       Class:  CommandQueue
 *      Method:  ~CommandQueue
 * Description:  destructor
 *------------------------------------------------------------------------------
 */
CommandQueue::~CommandQueue () {
    if (d_demand_distributions_ != NULL) {
        for (int i = 0; i < (int)device_params_->num_distri; ++i){
            cuda_FreeMemory(d_demand_distribution_[i]);
        }
        cuda_FreeMemory(device_params_->demand_distributions);
        free (d_demand_distributions_);
        device_params_->demand_distributions = NULL;
    }
}  /* -----  end of method CommandQueue::~CommandQueue  (destructor)  ----- */

/*
 *------------------------------------------------------------------------------
 *       Class:  CommandQueue
 *      Method:  operator =
 * Description:  assignment operator
 *------------------------------------------------------------------------------
 */
CommandQueue&
CommandQueue::operator = ( CommandQueue &other ) {
    if (this != &other){
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
std::string * CommandQueue::get_config_ptr (const char * var) {
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
const char * CommandQueue::get_config (const char * var) {
    if (get_config_ptr(var) == NULL){
        printf("Error: Cannot get the string of the %s.", var);
        return "";
    }
    return *get_config(var).c_str();
}       /* -----  end of method CommandQueue::get_config  ----- */

/*
 *------------------------------------------------------------------------------
 *       Class:  CommandQueue
 *      Method:  check_command
 * Description:  check whether we need to opearte certain commands
 *------------------------------------------------------------------------------
 */
bool CommandQueue::check_command (const char * var) {
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
 *      Method:  get_h_param
 * Description:  return the value stored in HostParameters
 *------------------------------------------------------------------------------
 */
float CommandQueue::get_h_param (const char * var) {
    return *host_params_[var];
}       /* -----  end of method CommandQueue::get_h_param  ----- */


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
bool CommandQueue::load_host_params ( const char * var, float value ) {
    if (host_params_->set_value(var,value)) {
        return true;
    }
    else return false;
}       /* -----  end of method CommandQueue::load_host_params  ----- */

/*
 *------------------------------------------------------------------------------
 *       Class:  CommandQueue
 *      Method:  load_files
 * Description:  load the parameters from the input file
 *------------------------------------------------------------------------------
 */
bool CommandQueue::load_files (const char * type) {
    if(type == "param"){
        FILE* fp = std::fopen(get_config("input_file_name"), "r");
        char readBuffer[65536];
        rapidjson::FileReadStream is(fp, readBuffer, sizeof(readBuffer));
        rapidjson::Document para;
        para.ParseStream(is);
        bool process_status = false;
        for (int i = 0; i < host_params_->get_param_num(); ++i){
            process_status = cmd->load_host_params(host_params_->pop_param_name(i),\
                    para[host_params_->pop_param_name(i)].GetDouble());
            if (!process_status){
                printf("Error: Failed to load parameter : %s, exit.",\
                        host_params_->pop_param_name(i));
                return false;
            }
        }
        /* :TODO:Sat Aug  1 12:33:28 2015:huangzonghao:
         *  how to get the host paramters
         */
        /* :TODO:Fri Aug  7 13:23:42 2015:huangzonghao:
         *  need to implemente the mulitiple distribution load in
         *  this is currently a temp solution
         */
        const rapidjson::Value& demand_array = para["demand_distribution"];
        for (int i = 0; i < host_params_->get_value("max_demand") -\
                host_params_->get_value("min_demand"); ++i){
            host_params_->set_distribution(0, i, demand_array[i].GetDouble());
        }

        fclose(fp);
        return true;
    }
}       /* -----  end of method CommandQueue::load_files  ----- */

/*
 *------------------------------------------------------------------------------
 *       Class:  CommandQueue
 *      Method:  update_device_params
 * Description:  passt the params stored in the HostParameters to
 *                 DeviceParameters
 *------------------------------------------------------------------------------
 */
bool CommandQueue::update_device_params () {
    device_params_->T            = (size_t)*host_params_["T"];
    device_params_->m            = (size_t)*host_params_["m"];
    device_params_->k            = (size_t)*host_params_["k"];
    device_params_->maxhold      = (size_t)*host_params_["maxhold"];
    device_params_->max_demand   = (size_t)*host_params_["max_demand"];
    device_params_->min_demand   = (size_t)*host_params_["min_demand"];
    device_params_->num_distri   = (size_t)*host_params_["num_distri"];
    device_params_->c            = *host_params_["c"];
    device_params_->h            = *host_params_["h"];
    device_params_->theta        = *host_params_["theta"];
    device_params_->r            = *host_params_["r"];
    device_params_->s            = *host_params_["s"];
    device_params_->alpha        = *host_params_["alpha"];
    device_params_->lambda       = *host_params_["lambda"];

    device_params_->table_length = pow(*host_params_["k"], *host_params_["m"]);

    /* now start to deal with the demand distribution */
    if (d_demand_distributions_ == NULL) {
        // assigning the memory to store the pointer to distributions
        d_demand_distributions_ =\
                        malloc(device_params_->num_distri * sizeof(float *));
        if (d_demand_distributions_ == NULL) {
            fprintf(stderr, "\ndynamic memory allocation failed\n");
            exit(EXIT_FAILURE);
        }
        // allocate the cuda memory for the distribution
        for (int i = 0; i < device_params_->num_distri; ++i){
            d_demand_distributions_[i] =\
                    cuda_AllocateMemoryFloat(device_params_->max_demand -\
                                        device_params_->min_demand);
        }
    }
    /* now pass the distribution to the device */
    for (int i = 0; i < device_params_->num_distri; ++i){
        cuda_PassToDevice(host_params_->get_distribution_ptr(i),\
                          device_params_->demand_distributions[i],\
                          device_params_->max_demand - device_params_->min_demand);
    }
    /* pass the holder of the distributions to device */
    device_params_->demand_distributions = cuda_AllocateMemoryFloatPtr\
                                            (device_params_->num_distri);
    cuda_PassToDevice(d_demand_distributions_,\
                      device_params_->demand_distributions,\
                      device_params_->num_distri);
    return true;
}       /* -----  end of method CommandQueue::update_device_params  ----- */

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
 *                         end of file command_queue.cc
 * =============================================================================
 */
