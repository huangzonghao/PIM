/*
 * =============================================================================
 *
 *       Filename:  support-inl.h
 *
 *    Description:  The definition of some supporting inline functions
 *
 *        Created:  Thu Jul 23 00:40:42 2015
 *       Modified:  Thu Jul 23 00:40:42 2015
 *
 *         Author:  Huang Zonghao
 *          Email:  coding@huangzonghao.com
 *
 * =============================================================================
 */
#ifndef SUPPORT-INL_H_
#define SUPPORT-INL_H_

/* #####   EXPORTED INCLINE FUNCTION DEFINE ################################## */

/*
 * ===  FUNCTION  ==============================================================
 *         Name:  DoesItExist
 *  Description:  Check the existence of a file
 *       @param:  filename
 *      @return:  whether the file exists or not
 * =============================================================================
 */
inline bool DoesItExist ( const char* filename ){
    ifstream f(name);
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
inline const char* ExeCMD ( const char * cmd ){
    FILE* pipe = popen(cmd, "r");
    if (!pipe) return "ERROR";
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



#endif   /* ----- #ifndef SUPPORT-INL_H_  ----- */

/* =============================================================================
 *                         end of file support-inl.h
 * =============================================================================
 */
