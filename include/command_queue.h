/*
 * =============================================================================
 *
 *       Filename:  command_queue.h
 *
 *    Description:  This file contains the classes holding all the configuration
 *                    parameters and controlling information
 *
 *        Created:  Thu Jul 23 00:45:56 2015
 *       Modified:  Fri Aug  7 00:49:29 2015
 *
 *         Author:  Huang Zonghao
 *          Email:  coding@huangzonghao.com
 *
 * =============================================================================
 */
#ifndef COMMAND_QUEUE_H_
#define COMMAND_QUEUE_H_
#include <stdlib.h>
#include <string>

class HostParameters;
class DeviceParameters;

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
    HostParameters * get_host_param_pointer();
    DeviceParameters * get_device_param_pointer();
    float get_host_param_value(const std::string &var);
    std::string get_config(const std::string &var);


    bool check_command(const std::string &var);
    bool check_status(const std::string &var);


    /* =========================   MUTATORS    =============================== */
    bool load_host_params(std::string var, float value);
    /*
     * FLAGS : INPUTFILE | OUTPUTFILE | OUPUTFORMAT | POLICY | RECOVERY |
     *            ENABLEVERBOSE | ENABLELOG | PRINTHELP | RECORDING
     */
    bool load_commands( const std::string var, const std::string value );

    /* =========================   OPERATORS   =============================== */
    bool update_device_params();
    bool retrieve_device_params();
    /* assignment operator */
    CommandQueue& operator = ( const CommandQueue &other );

    /* print out the parameters stored in HostParameters */
    void print_params();
  private:
    /* ========================  DATA MEMBERS  =============================== */

    HostParameters * host_params_;
    DeviceParameters * device_params_;

    /* Config List
     * No.  type    name
     * 0    string  input_file_name
     * 1    string  output_file_name
     * 2    string  output_format
     * 3    string  policy
     * 4    string  recovery_file_name
     * 5    string  recording_file_name
     * 6    string  logging_file_name
     */
    const int num_configs_ = 7;
    std::string configs_[7];
    const char* config_names_[7] = {"input_file_name",
                                    "output_file_name",
                                    "output_format",
                                    "policy",
                                    "recovery_file_name",
                                    "recording_file_name",
                                    "logging_file_name"};

    bool verbose_enabled_;
    bool recovery_enabled_;
    bool logging_enabled_;
    bool recording_enabled_;
    bool print_help_;

    /* State Values */
    bool commands_loaded_;
    bool parameters_loaded_;

    std::string * get_config_ptr(const std::string &var);

}; /* -----  end of class CommandQueue  ----- */




#endif   /* ----- #ifndef COMMAND_QUEUE_H_  ----- */


/* =============================================================================
 *                         end of file command_queue.h
 * =============================================================================
 */
