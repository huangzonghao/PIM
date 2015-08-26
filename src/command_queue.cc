/*
 * =============================================================================
 *
 *       Filename:  command_queue.cc
 *
 *    Description:  This file contains the implementation of CommandQueue
 *
 *        Created:  Fri Jul 24 13:52:37 2015
 *       Modified:  Mon Aug 17 09:22:39 2015
 *
 *         Author:  Huang Zonghao
 *          Email:  coding@huangzonghao.com
 *
 * =============================================================================
 */
#include "../include/command_queue.h"

#include <fstream>
#include <stdlib.h>
#include <math.h>

#include "../thirdparty/rapidjson/document.h"
#include "../thirdparty/rapidjson/prettywriter.h"
#include "../thirdparty/rapidjson/filereadstream.h"
#include "../thirdparty/rapidjson/filewritestream.h"

#include "../include/support-inl.h"
#include "../include/host_parameters.h"
#include "../include/device_parameters.h"
#include "../include/cuda_support.h"

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
            configs_[i] = other.configs_[i];
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
    /* the dynamic memory allocations that needed to be cleaned up:
     *     1) host_params_ and device_params_ on host memory
     *     2) demand distirbution tables on the device memory
     *     3) the pointer holding the distirbution tables on the device memory
     */

    /* first clean 2 */
    if(!demand_table_pointers.size()){
        for (int i = 0; i < (int)demand_table_pointers.size(); ++i){
            cuda_FreeMemory(demand_table_pointers[i]);
        }
        /* then 3 */
        cuda_FreeMemory(device_params_->demand_distributions);
        device_params_->demand_distributions = NULL;
    }
    /* finally 1 */
    delete host_params_;
    delete device_params_;
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
    /* components need to copy:
     *     1) host_params_ and device_params_
     *     2) demand_table_pointers
     *     3) configs_
     *     4) 5 enables
     */
    if (this != &other){
        *host_params_ = *other.get_host_param_pointer();
        update_device_params();
        demand_table_pointers = other.demand_table_pointers;
        for (int i = 0; i < num_configs_; ++i){
            configs_[i] = other.configs_[i];
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
        if (strcmp(var, config_names_[i]) == 0)
            return configs_ + i;
    }
    printf("Error: %s is not a config variable name, return NULL.\n", var);
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
        printf("Error: Cannot get the string of the %s.\n", var);
        return "";
    }
    return get_config_ptr(var)->c_str();
}       /* -----  end of method CommandQueue::get_config  ----- */

/*
 *------------------------------------------------------------------------------
 *       Class:  CommandQueue
 *      Method:  check_command
 * Description:  check whether we need to opearte certain commands
 *------------------------------------------------------------------------------
 */
bool CommandQueue::check_command (const char * var) {
        if( strcmp(var, "verbose") == 0)
            return verbose_enabled_;
        if( strcmp(var, "recovery") == 0)
            return recovery_enabled_;
        if( strcmp(var, "log") == 0)
            return logging_enabled_;
        if( strcmp(var, "record") == 0)
            return recording_enabled_;
        if( strcmp(var, "print_help") == 0)
            return print_help_;
        else{
            printf("Error: Invalide command name, check_command failed.\n");
            exit(-1);
        }
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
    return host_params_->get_value(var);
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
    if (host_params_->set_param(var,value)) {
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
    if(strcmp(type, "param") == 0){
        FILE *fp = std::fopen(get_config("input_file_name"), "r");
        char readBuffer[65536];
        rapidjson::FileReadStream is(fp, readBuffer, sizeof(readBuffer));
        rapidjson::Document para;
        para.ParseStream(is);
        bool process_status = false;
        for (int i = 0; i < host_params_->get_param_num(); ++i){
            process_status = host_params_->set_param(host_params_->pop_param_names(i),\
                    para[host_params_->pop_param_names(i)].GetDouble());
            if (!process_status){
                printf("Error: Failed to load parameter : %s, exit.\n",\
                        host_params_->pop_param_names(i));
                return false;
            }
        }
        /* :TODO:Sat Aug  1 12:33:28 2015:huangzonghao:
         *  how to get the host paramters
         */
/*-----------------------------------------------------------------------------
 *  bookmark!!!!!
 *-----------------------------------------------------------------------------*/
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
    printf("Error: %s is not a valid file type, exit\n", type);
    return false;
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
    device_params_->T            = (size_t)host_params_->get_value("T");
    device_params_->m            = (size_t)host_params_->get_value("m");
    device_params_->k            = (size_t)host_params_->get_value("k");
    device_params_->maxhold      = (size_t)host_params_->get_value("maxhold");
    device_params_->max_demand   = (size_t)host_params_->get_value("max_demand");
    device_params_->min_demand   = (size_t)host_params_->get_value("min_demand");
    device_params_->num_distri   = (size_t)host_params_->get_value("num_distri");
    device_params_->c            = host_params_->get_value("c");
    device_params_->h            = host_params_->get_value("h");
    device_params_->theta        = host_params_->get_value("theta");
    device_params_->r            = host_params_->get_value("r");
    device_params_->s            = host_params_->get_value("s");
    device_params_->alpha        = host_params_->get_value("alpha");
    device_params_->lambda       = host_params_->get_value("lambda");

    device_params_->table_length = pow(host_params_->get_value("k"), host_params_->get_value("m"));

    /* now start to deal with the demand distribution */
    if (demand_table_pointers.empty()){
        // allocate the cuda memory for the distribution
        for (int i = 0; i < (int)device_params_->num_distri; ++i){
            demand_table_pointers.push_back(cuda_AllocateMemoryFloat(\
                        device_params_->max_demand - device_params_->min_demand));
        }
    }
    else if ( demand_table_pointers.size() != device_params_->num_distri ){
        /* adjust the number of pointers to what we want */
        if ( demand_table_pointers.size() > device_params_->num_distri ){
            for (int i = 0; i < (int)(demand_table_pointers.size() - device_params_->num_distri); ++i){
                cuda_FreeMemory(demand_table_pointers.back());
                demand_table_pointers.pop_back();
            }
        }
        else {
            for (int i = 0; i < (int)(device_params_->num_distri - demand_table_pointers.size()); ++i){
            demand_table_pointers.push_back(cuda_AllocateMemoryFloat(\
                        device_params_->max_demand - device_params_->min_demand));
            }
        }
    }

    /* now we are sure that the numnber of cantainers and the number of distribution
     * tables are the same
     * now pass the distribution to the device
     *
     * steps:
     *     1) pass the distribution tables to the device
     *     2) create the on device containers for the pointers
     *     3) pass the pointers to the device
     */
    for (int i = 0; i < (int)device_params_->num_distri; ++i){
        cuda_PassToDevice(host_params_->get_distribution_ptr(i),\
                          demand_table_pointers[i],\
                          device_params_->max_demand - device_params_->min_demand);
    }
    device_params_->demand_distributions =
        cuda_AllocateMemoryFloatPtr(device_params_->num_distri);
    for (int i = 0; i < demand_table_pointers.size(); ++i){
        cuda_PassToDevice
    }

    return true;
}       /* -----  end of method CommandQueue::update_device_params  ----- */

/*
 *------------------------------------------------------------------------------
 *       Class:  CommandQueue
 *      Method:  load_commands
 * Description:  set the control parameters correspondingly
 *------------------------------------------------------------------------------
 */
bool CommandQueue::load_commands (const char *var, const char *value) {
        if( strcmp(var, "INPUT_FILE") == 0 )
            *get_config_ptr("input_file_name") = value;
        if( strcmp(var, "OUPUT_FILE") == 0 )
            *get_config_ptr("output_file_name") = value;
        if( strcmp(var, "OUPUT_FORMAT") == 0 ){
            if (!IsValidFileFormat(value)){
                printf("Error: Invalid file format, exit\n");
                return false;
            }
            *get_config_ptr("output_file_format") = value;
        }
        if( strcmp(var, "POLICY") == 0 ){
            if (!IsValidPolicy(value)){
                printf("Error: Invalid policy, exit\n");
                return false;
            }
            *get_config_ptr("policy") = value;
        }
        if( strcmp(var, "RECOVERY") == 0 ){
            *get_config_ptr("recovery_file_name") = value;
            recovery_enabled_ = true;
        }
        if( strcmp(var, "RECORDING") == 0 ){
            *get_config_ptr("recording_file_name") = value;
            recording_enabled_ = true;
        }
        if( strcmp(var, "ENABLE_VERBOSE") == 0 )
            verbose_enabled_ = true;
        if( strcmp(var, "LOGGING") == 0 ){
            *get_config_ptr("logging_file_name") = value;
            logging_enabled_ = true;
        }
        if( strcmp(var, "PRINT_HELP") == 0 )
            print_help_ = true;
        else {
            printf("%s is not a valid attribute of CommandQueue, exit.\n", var);
            return false;
        }
    return true;
}       /* -----  end of method CommandQueue::load_commands  ----- */


/* =============================================================================
 *                         end of file command_queue.cc
 * =============================================================================
 */
