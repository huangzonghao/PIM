/*
 * =============================================================================
 *
 *       Filename:  command_queue.h
 *
 *    Description:  This file contains the classes holding all the configuration
 *                    parameters and controlling information
 *
 *        Created:  Thu Jul 23 00:45:56 2015
 *       Modified:  Thu 10 Sep 2015 09:07:41 AM HKT
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
#include <vector>

class HostParameters;
struct DeviceParameters;
struct DemandDistribution;

/*
 * =============================================================================
 *        Class:  CommandQueue
 *  Description:  This class contains all the configuration and control
 *                  information of the task.
 *                And note this class should not be involved into the real
 *                caltulation. It is just a database holding all the
 *                parameters and addresses.
 * =============================================================================
 */
class CommandQueue
{
  public:
    /* =========================   LIFECYCLE   =============================== */

    /* constructor */
    CommandQueue ();
    /* copy constructor */
    CommandQueue ( CommandQueue &other );
    /* destructor */
    ~CommandQueue ();

    /* =========================   ACCESSORS   =============================== */
    HostParameters *get_host_param_pointer();
    DeviceParameters *get_device_param_pointer();
    DeviceParameters get_device_param_struct();
    struct DemandDistribution *get_h_demand_pointer(int index);
    float get_h_param(const char *var);
    float get_d_param(const char *var);
    const char *get_config(const char *var);

    bool check_command(const char *var);

    /* =========================   MUTATORS    =============================== */
    bool load_host_params(const char *var, float value);
    bool load_files(const char *type);
    /*
     * FLAGS : INPUTFILE | OUTPUTFILE | OUPUTFORMAT | POLICY | RECOVERY |
     *            ENABLEVERBOSE | ENABLELOG | PRINTHELP | RECORDING
     */
    bool load_commands( const char *var, const char *value );

    /* =========================   OPERATORS   =============================== */
    bool update_device_params();
    /* assignment operator */
    CommandQueue& operator = ( CommandQueue &other );

    /* print out the parameters stored in HostParameters */
    void print_params();
  private:
    /* ========================  DATA MEMBERS  =============================== */

    HostParameters *host_params_;
    /* note when passing to kernel, we shall not use pass by reference anymore */
    struct DeviceParameters *device_params_;

    /* :REMARKS:Mon Aug 10 23:17:52 2015:huangzonghao:
     *  when in .cc file of course we are gonna use std containers to manage
     *  the dynamic stuff.....
     */
    /* this holds the pointers to the distribution table on the device */
    std::vector<DemandDistribution*> demand_table_pointers;

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
    static const int num_configs_ = 7;

    /* :REMARKS:Fri Aug  7 13:08:20 2015:huangzonghao:
     *  Use string here just for the simplicity to manage dynamic string length
     *  shall prefer char array when dealing with const chars
     */
    std::string configs_[7];

    /* you don't have to indicate the array size when initialization, but you have
     * to fix the array size here! at decalaration!
     */
    static const char *config_names_[7];

    bool verbose_enabled_;
    bool recovery_enabled_;
    bool logging_enabled_;
    bool recording_enabled_;
    bool print_help_;

    std::string *get_config_ptr(const char *var);

}; /* -----  end of class CommandQueue  ----- */


#endif   /* ----- #ifndef COMMAND_QUEUE_H_  ----- */
/* =============================================================================
 *                         end of file command_queue.h
 * =============================================================================
 */
