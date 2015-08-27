/*
 * =============================================================================
 *
 *       Filename:  support.cc
 *
 *    Description:  The functions in this file are for general work flow of the
 *                    task and have nothing to do with the specific computation
 *                    or algorithm
 *
 *        Created:  Wed Jul 22 14:11:43 2015
 *       Modified:  Thu Aug 27 16:34:02 2015
 *
 *         Author:  Huang Zonghao
 *          Email:  coding@huangzonghao.com
 *
 * =============================================================================
 */

/* #####   HEADER FILE INCLUDES   ############################################ */
#include "../include/support.h"

#include <stdlib.h>
#include <unistd.h>
#include <signal.h>
#include <fstream>
#include <iostream>

#include "../include/support-inl.h"
#include "../include/command_queue.h"

/* the following macros are for the format of the progress recording file */
#define PROGRESS_RECORDING_FILE_HEADER "/* Progress Recording File */"
#define PROGRESS_RECORDING_FILE_FIRST_TABLE "/* Table To Update */"
#define PROGRESS_RECORDING_FILE_SECOND_TABLE "/* Table For Reference */"

/* #####   FUNCTION DEFINITIONS  -  EXPORTED FUNCTIONS   ##################### */

/*
 * ===  FUNCTION  ==============================================================
 *         Name:  PrintUsage
 *  Description:  prints the usages of the entire program
 *       @param:  none
 *      @return:  none
 * =============================================================================
 */
void PrintUsage (){
    printf( "This program solves the perishable inventory problem.\n"
            "Usage:\n"
            "\t-i <filename>\n"
            "\t\tSet the input parameter file name, default = params.json\n"
            "\t-o <filename>\n"
            "\t\tSet the output file name, default = output.txt\n"
            "\t-f [csv | nature | json | xml]\n"
            "\t\tSet the output file format, default = csv\n"
            "\t-p [all | tree | fluid]\n"
            "\t\tSet the inventory management policy, default = all\n"
            "\t-r <filename>\n"
            "\t\tRestart from the previous ending point, the save file must be\n"
            "\t\tindicated. This option will disable all the other settings\n"
            "\t-s <filename>\n"
            "\t\tPeriodically record the progress of the program. All the data\n"
            "\t\twill be write to the indicaded file\n"
            "\t-l <filename>\n"
            "\t\tWrite task log to the indicated file\n"
            "\t-v\n"
            "\t\tVerbose, print out all the task information\n"
            "\t-h\n"
            "\t-?\n"
            "\t\tPrint this help manual\n"
            );
    return;
}       /* -----  end of function PrintUsage  ----- */

/*
 * ===  FUNCTION  ==============================================================
 *         Name:  LoadCommands
 *  Description:  This function loads and parses the commands from bash and sets
 *                  sets the global control variabls correspondingly
 *       @param:  argc, agrv and the command queue
 *      @return:  the loading status
 * =============================================================================
 */
 /* :TODO:Thu Jul 23 00:38:06 2015 00:38:huangzonghao:
  * need to develope structure containing all the task configuration information
  * and this structure shall be the same as the one we use in recording
  */
extern char *optarg;
bool LoadCommands ( int argc, char ** argv, CommandQueue * cmd ){
    if (argc < 2){
        printf("Insufficient input, checkout the usage:\n");
        PrintUsage();
        return false;
    }
    char command;
    while ((command = getopt(argc, argv, "?i:o:f:p:r:s:l:vh")) > 0){
        switch (command){
            case 'i':
                break;
            case 'o':
                cmd->load_commands("OUTPUT_FILE", optarg);
                break;
            case 'f':
                if(IsValidFileFormat(optarg)){
                    cmd->load_commands("OUTPUT_FORMAT", optarg);
                }
                else {
                    printf("Invalid output format, exit");
                    return false;
                }
                break;
            case 'p':
                if(IsValidPolicy(optarg)){
                    cmd->load_commands("POLICY", optarg);
                }
                else {
                    printf("Invalid policy, exit");
                    return false;
                }
                break;
            case 'r':
                cmd->load_commands("RECOVERY", optarg);
                break;
            case 's':
                cmd->load_commands("RECORD", optarg);
                break;
            case 'l':
                cmd->load_commands("LOGGING", optarg);
                break;
            case 'v':
                cmd->load_commands("ENABLE_VERBOSE", "1");
                break;
            case '?':
            case 'h':
                cmd->load_commands("PRINT_HELP", "1");
                break;
        }
    }

    return true;
}       /* -----  end of function LoadCommands  ----- */


/*
 * ===  FUNCTION  ==============================================================
 *         Name:  InterruptHandler
 *  Description:  To catch the <C-C> event and wrap up the progress before
 *                  exit.
 *       @param:  s: event type
 *      @return:  none
 * =============================================================================
 */
 /* :TODO:Wed Jul 22 17:03:32 2015 17:03:huangzonghao: when exiting, first report
  * the current progress, then ask if need to store the current progress */
/* void InterruptHandler ( int s ){
 *     if (s == 2){
 *         printf("User interrupt! Exit\n");
 *         return;
 *     }
 * }       [> -----  end of function InterruptHandler  ----- <]
 */


/*
 * ===  FUNCTION  ==============================================================
 *         Name:  LoadParameters
 *  Description:  This function will search for the input file whose name is
 *                  stored in the CommandQueue and load the parameters stored
 *                  in it to the HostParameters struct
 *       @param:  input_filename: the filename of the json file
 *      @return:  the status of loading
 * =============================================================================
 */

 /* :TODO:Wed Jul 22 17:17:20 2015 17:17:huangzonghao:
  * if all the parameters successfully loaded, return true
  * if either 1) the file not found
  *           2) the file is not complete
  * then print the error msg and return false
  */
bool LoadParameters ( CommandQueue * cmd ){
    if(!DoesItExist(cmd->get_config("input_file_name"))){
        printf("Error: Cannot find file %s", cmd->get_config("input_file_name"));
        return false;
    }
    bool processing_status = false;
    processing_status = cmd->load_files("param");
    if (processing_status){
        return true;
    }
    else {
        printf("Error: something went wrong while loading the parameters");
        return false;
    }
}       /* -----  end of function LoadParameters  ----- */


/*
 * ===  FUNCTION  ==============================================================
 *         Name:  WriteOutputFile
 *  Description:  Convert the result to specific format and write to the output
 *                    file
 *       @param:  pointer_to_value_table, format, filename
 *      @return:  output status
 * =============================================================================
 */
 /* :TODO:Wed Jul 22 17:32:48 2015 17:32:huangzonghao:
  * first check if the output file exists, if so, attach the time stamp to the
  * output file name and print the error msg
  * then generate the output file based on the indicated format and return status
  */
bool WriteOutputFile ( const float *value_table,
                       const size_t table_length,
                       const int output_format,
                       const char *output_file_name ){

    if ( DoesItExist(output_file_name) ) {
        std::string user_option;
        printf("%s already exists, overwritten? (y/n) : ", output_file_name);
        std::cin >> user_option;
        if ( user_option == "y")
            remove(output_file_name);
        else {
            return false;
        }
    }
    std::ofstream ofs;
    ofs.open (output_file_name, std::ofstream::out | std::ofstream::app);
    switch(output_format){
        case 1:
            for(size_t i = 0; i < table_length; ++i){
                ofs << value_table[i] << std::endl;
            }
    }

    return true;
}       /* -----  end of function WriteOutputFile  ----- */


/*
 * ===  FUNCTION  ==============================================================
 *         Name:  RecordProgress
 *  Description:  Store the current progress to a backup file
 *       @param:  pointer_to_two_value_tables, record_label
 *      @return:  status of recording
 * =============================================================================
 */

/* :REMARKS:Thu Aug 27 13:48:06 2015:huangzonghao:
 *  Note when ever we want to use the recording functionality, the corresponding
 *  configuration file should always be there, so we can just load the progress
 *  before the real calculation
 */

 /* :TODO:Wed Jul 22 17:41:14 2015 17:41:huangzonghao:
  * a zero length record_label means this is the first time to store, then create
  * the file and name it after the timestamp then store the name to record_label
  * if the record_table is not empty, then wirte the new progress to a temp file
  * first then rename it to the record_label
  * May need to set up a struct to store some status values
  */
bool RecordProgress ( const float *first_table,
                      const float *second_table,
                      const size_t table_length,
                      const char *progress_file_name ){
    printf("Now strat to record the progress.\n");
    if ( DoesItExist(progress_file_name) ) {
        std::string user_option;
        printf("The file %s already exists, overwirtten? (y/n) : ", progress_file_name);
        std::cin >> user_option;
        if ( user_option == "y")
            remove(progress_file_name);
        else {
            return false;
        }
    }
    std::ofstream ofs;
    ofs.open (progress_file_name , std::ofstream::out | std::ofstream::app);
    /* write the header */
    ofs << PROGRESS_RECORDING_FILE_HEADER << std::endl;
    /* first write the current table */
    ofs << PROGRESS_RECORDING_FILE_FIRST_TABLE << std::endl;
    for(size_t i = 0; i < table_length; ++i){
        ofs << first_table[i] << std::endl;
    }

    /* then the table to update */
    ofs << PROGRESS_RECORDING_FILE_SECOND_TABLE << std::endl;
    for (size_t i = 0; i < table_length; ++i){
        ofs << second_table[i] << std::endl;
    }

    ofs.close();
    printf("The progerss recording file %s has written successfully!\n", progress_file_name);

    return true;
}       /* -----  end of function RecordProgress  ----- */


/*
 * ===  FUNCTION  ==============================================================
 *         Name:  LoadProgress
 *  Description:  Load the previous stored recording file to the provided two host
 *                  arrays which will be handled by the functions in frame
 *       @param:  recording_file_name, pointer_to_two_value_tables
 *      @return:  status of loading
 * =============================================================================
 */
bool LoadProgress ( float *first_table,
                    float *second_table,
                    const size_t table_length,
                    const char *progress_file_name ){
    std::ifstream fin (progress_file_name);
    if (!fin.is_open()){
        printf("Error: The progerss recording file %s is not found!\n", progress_file_name);
        return false;
    }
    std::string temp_line;

    /* get the first line and check if this is a progerss recording file */
    std::getline(fin, temp_line);
    if(temp_line != PROGRESS_RECORDING_FILE_HEADER){
        printf("Error: %s is not a valid progress recording file.!\n", progress_file_name);
        return false;
    }

    /* then the first table */
    std::getline(fin, temp_line);
    if(temp_line != PROGRESS_RECORDING_FILE_FIRST_TABLE){
        printf("Error: cannot find the information of the first table!\n");
        return false;
    }
    for (size_t i = 0; i < table_length; ++i){
        fin >> first_table[i];
    }

    /* then the second table */
    std::getline(fin, temp_line);
    if(temp_line != PROGRESS_RECORDING_FILE_SECOND_TABLE){
        printf("Error: cannot find the information of the second table!\n");
        return false;
    }
    for(size_t i = 0; i < table_length; ++i){
        fin >> second_table[i];
    }
    fin.close();
    return true;
}       /* -----  end of function LoadProgress  ----- */


/*
 * ===  FUNCTION  ==============================================================
 *         Name:  PrintVerboseInfo
 *  Description:  Print the verbose information of current task
 *       @param:  don't know yet
 *      @return:  don't know....
 * =============================================================================
 */
 /* :TODO:Wed Jul 22 17:52:42 2015 17:52:huangzonghao:
  * the function will be called at the end of each loop, print the info of current
  * progress
  */
/* void PrintVerboseInfo (){
 *
 *     return;
 * }       [> -----  end of function PrintVerboseInfo  ----- <]
 */


/*
 * ===  FUNCTION  ==============================================================
 *         Name:  WriteLog
 *  Description:  write the timeline based log to the given file
 *       @param:  <+PARAMETERS+>
 *      @return:  <+RETURN_VALUES+>
 * =============================================================================
 */
 /* :TODO:Thu Jul 23 01:31:48 2015 01:31:huangzonghao:
  * need to use some flag to decided which position the function is at, the starting
  * of the program or the middle or what....
  */
/* bool WriteLog ( CommandQueue &cmd ){
 *     cmd->print_log();
 *     return true;
 * }       [> -----  end of function WriteLog  ----- <]
 */


/* =============================================================================
 *                         end of file support.cc
 * =============================================================================
 */
