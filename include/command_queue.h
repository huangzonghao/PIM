/*
 * =============================================================================
 *
 *       Filename:  command_queue.h
 *
 *    Description:  This file contains the classes holding all the configuration
 *                    parameters and controlling information
 *
 *        Created:  Thu Jul 23 00:45:56 2015
 *       Modified:  Sat Aug  1 12:43:51 2015
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
    HostParameters * get_host_param_pointer();
    DeviceParameters * get_device_param_pointer();
    float get_host_param_value();

    bool do_printing_help();
    bool do_verbose();
    const char * get_input_file_name();
    const char * get_output_file_name();
    const char * get_output_format();
    const char * get_policy();
    const char * get_recovery_file_name();
    const char * get_logging_file_name();
    const char * get_recording_file_name();


    /* =========================   MUTATORS    =============================== */
    bool load_host_params(const char * var, float value);
    /*
     * FLAGS : INPUTFILE | OUTPUTFILE | OUPUTFORMAT | POLICY | RECOVERY |
     *            ENABLEVERBOSE | ENABLELOG | PRINTHELP | RECORDING
     */
    bool load_commands( const char * var, const char * value );

    /* =========================   OPERATORS   =============================== */
    bool update_device_params();
    bool retrieve_device_params();
    /* assignment operator */
    CommandQueue& operator = ( const CommandQueue &other );

    /* print out the parameters stored in HostParameters */
    void print_params();
  private:
    /* ========================  DATA MEMBERS  =============================== */

    HostParameters h;
    DeviceParameters d;
    std::string input_file_name_  = "params.json";
    std::string output_file_name_ = "output.txt";
    std::string output_format_    = "csv";
    std::string policy_           = "all";
    std::string recovery_file_    = "";
    std::string loggirg_file_     = "";
    std::string recording_file_   = "";
    bool is_verbose_enabled_      = false;
    bool is_recovery_enabled_     = false;
    bool is_logging_enabled_      = false;
    bool is_recording_enabled_    = false;
    bool print_help_              = false;

}; /* -----  end of class CommandQueue  ----- */




#endif   /* ----- #ifndef COMMAND_QUEUE_H_  ----- */


/* =============================================================================
 *                         end of file command_queue.h
 * =============================================================================
 */
