/*
 * =============================================================================
 *
 *       Filename:  support.cc
 *
 *    Description:  This file contains the supporting tools for the porject
 *                    but no algorithm functions here
 *
 *        Created:  Wed Jul 22 14:11:43 2015
 *       Modified:  Wed Jul 22 14:11:43 2015
 *
 *         Author:  Huang Zonghao
 *          Email:  coding@huangzonghao.com
 *
 * =============================================================================
 */

/* #####   HEADER FILE INCLUDES   ############################################ */
#include "include/support.h"
#include <stdlib.h>
#include <unistd.h>
#include <signal.h>
#include <string.h>
#include <fstream>
#include "thirdparty/rapidjson/document.h"
#include "thirdparty/rapidjson/prettywriter.h"
#include "thirdparty/rapidjson/filereadstream.h"
#include "thirdparty/rapidjson/filewritestream.h"
#include "include/support-inl.h"

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
bool LoadCommands ( int argc, char ** argv, CommandQueue cmd ){

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
void InterruptHandler ( int s ){

    return ;
}       /* -----  end of function InterruptHandler  ----- */


/*
 * ===  FUNCTION  ==============================================================
 *         Name:  LoadParameters
 *  Description:  Load the parameters from the json file and valid the input data
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
bool LoadParameters ( const char* input_filename ){

    return true;
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
bool WriteOutputFile ( const float * value_table, const char * output_format,\
                       const char * output_filename ){

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
 /* :TODO:Wed Jul 22 17:41:14 2015 17:41:huangzonghao:
  * a zero length record_label means this is the first time to store, then create
  * the file and name it after the timestamp then store the name to record_label
  * if the record_table is not empty, then wirte the new progress to a temp file
  * first then rename it to the record_label
  * May need to set up a struct to store some status values
  */
bool RecordProgress ( const float * current_value_table,\
                      const float * prev_value_table,\
                      const char * record_label ){

    return true;
}       /* -----  end of function RecordProgress  ----- */


/*
 * ===  FUNCTION  ==============================================================
 *         Name:  LoadProgress
 *  Description:  Load the previous stored recording file
 *       @param:  recording_file_name, pointer_to_two_value_tables
 *      @return:  status of loading
 * =============================================================================
 */
 /* :TODO:Wed Jul 22 17:51:56 2015 17:51:huangzonghao:
  * print the error msg if the recording file is not found and return false
  */
bool LoadProgress ( const char * record_filename,\
                    const float * current_value_table,\
                    const float * prev_value_table ){

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
void PrintVerboseInfo (){

    return;
}       /* -----  end of function PrintVerboseInfo  ----- */


