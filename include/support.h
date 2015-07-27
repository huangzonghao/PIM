/*
 * =============================================================================
 *
 *       Filename:  support.h
 *
 *    Description:  This is the header file of support.h
 *
 *        Created:  Wed Jul 22 18:38:32 2015
 *       Modified:  Mon Jul 27 19:52:09 2015
 *
 *         Author:  Huang Zonghao
 *          Email:  coding@huangzonghao.com
 *
 * =============================================================================
 */

#ifndef SUPPORT_H_
#define SUPPORT_H_
#include "command_queue.h"

/* #####   EXPORTED FUNCTION DECLARATIONS   ################################## */

void PrintUsage ();

void InterruptHandler ( int s );

bool LoadParameters ( const char* input_filename );

bool LoadCommands (const int argc, const char ** argv, CommandQueue * control);

bool WriteOutputFile ( const float * value_table,\
                       const char * output_format,\
                       const char * output_filename );

bool RecordProgress ( const float * current_value_table,\
                       const float * prev_value_table,\
                       const char * record_label );

bool LoadProgress ( const char * record_filename,\
                    const float * current_value_table,\
                    const float * prev_value_table );

void PrintVerboseInfo ();


#endif   /* ----- #ifndef SUPPORT_H_  ----- */

/* =============================================================================
 *                         end of file support.h
 * =============================================================================
 */
