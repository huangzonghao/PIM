/*
 * =============================================================================
 *
 *       Filename:  support-inl.h
 *
 *    Description:  The definition of some supporting inline functions
 *
 *        Created:  Thu Jul 23 00:40:42 2015
 *       Modified:  Thu 24 Sep 2015 03:14:49 AM HKT
 *
 *         Author:  Huang Zonghao
 *          Email:  coding@huangzonghao.com
 *
 * =============================================================================
 */
#ifndef SUPPORT_INL_H_
#define SUPPORT_INL_H_

#include "support.h"

#include <string.h>

#include <fstream>
#include "models.h"

/* #####   EXPORTED INCLINE FUNCTION DEFINE ################################## */

/*
 * ===  FUNCTION  ==============================================================
 *         Name:  DoesItExist
 *  Description:  Check the existence of a file
 *       @param:  filename
 *      @return:  whether the file exists or not
 * =============================================================================
 */
inline bool DoesItExist ( const char *filename ){
    std::ifstream f(filename);
    if (f.good()) {
        f.close();
        return true;
    }
    else {
        f.close();
        return false;
    }
}       /* -----  end of function DoesItExist  ----- */



/*
 * ===  FUNCTION  ==============================================================
 *         Name:  ExeCMD
 *  Description:  Execute some system commands and return the stdout
 *       @param:  sys_command
 *      @return:  output of stdout
 * =============================================================================
 */
inline const char *ExeCMD ( const char *cmd ){
    FILE *pipe = popen(cmd, "r");
    if (!pipe) return "ERROR : Cannot execute command";
    char buffer[128];
    std::string result = "";
    while(!feof(pipe)) {
        if(fgets(buffer, 128, pipe) != NULL)
            result += buffer;
    }
    pclose(pipe);
    result.resize(result.size() - 1);
    return result.c_str();
}       /* -----  end of function ExeCMD  ----- */

/*
 * ===  FUNCTION  ==============================================================
 *         Name:  IsValidFileFormat
 *  Description:  Check whether a string is the vaild output file format
 *       @param:  char *
 *      @return:  bool
 * =============================================================================
 */
inline bool IsValidFileFormat(const char *var){
    for (int i = 0; i < num_file_format_options; ++i){
        if (strcmp(var, file_format_options[i]) == 0)
            return true;
    }
    return false;
}       /* -----  end of function IsValidFileFormat  ----- */

/*
 * ===  FUNCTION  ==============================================================
 *         Name:  IsValidPolicy
 *  Description:  check whether the input string is the valid policy
 *       @param:  char *
 *      @return:  bool
 * =============================================================================
 */
inline bool IsValidPolicy(const char *var){
    for (int i = 0; i < num_policy_options; ++i){
        if (strcmp(var, policy_options[i]) == 0)
            return true;
    }
    return false;
}       /* -----  end of function IsValidPolicy  ----- */


#endif   /* ----- #ifndef SUPPORT-INL_H_  ----- */

/* =============================================================================
 *                         end of file support-inl.h
 * =============================================================================
 */
