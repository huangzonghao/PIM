/*
 * =============================================================================
 *
 *       Filename:  support.h
 *
 *    Description:  This is the header file of support.h
 *
 *        Created:  Wed Jul 22 18:38:32 2015
 *       Modified:  Wed Jul 22 18:38:32 2015
 *
 *         Author:  Huang Zonghao
 *          Email:  coding@huangzonghao.com
 *
 * =============================================================================
 */

#ifndef SUPPORT_H_
#define SUPPORT_H_

/* #####   FUNCTION DEFINITIONS  -  EXPORTED FUNCTIONS   ##################### */
void PrintUsage ();

void InterruptHandler ( int s );

bool LoadParameters ( const char* input_filename );

bool WriteOutputFile ( const float * value_table, const char * output_format,\
                       const char * output_filename );

bool RecordProgress ( const float * current_value_table,\
                       const float * prev_value_table,\
                       const char * record_label );

bool LoadProgress ( const char * record_filename,\
                    const float * current_value_table,\
                    const float * prev_value_table )

void PrintVerboseInfo ();

/* #####   FUNCTION DEFINITIONS  -  LOCAL TO THIS SOURCE FILE   ############## */
inline bool DoesItExist ( const char* filename );

const char* ExeCMD ( const char * cmd );

#endif   /* ----- #ifndef SUPPORT_H_  ----- */
