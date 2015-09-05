/*
 * =============================================================================
 *
 *       Filename:  support.h
 *
 *    Description:  This is the header file of support.h
 *
 *        Created:  Wed Jul 22 18:38:32 2015
 *       Modified:  Sat Sep  5 14:45:23 2015
 *
 *         Author:  Huang Zonghao
 *          Email:  coding@huangzonghao.com
 *
 * =============================================================================
 */

#ifndef SUPPORT_H_
#define SUPPORT_H_
#include <string>

class CommandQueue;

/* the supported output file formats */
const int num_file_format_options = 4;
const char *file_format_options[] = { "csv",
                                       "nature",
                                       "json",
                                       "xml" };
/* #####   EXPORTED FUNCTION DECLARATIONS   ################################## */

void PrintUsage();

void InterruptHandler( int s );

bool LoadParameters(CommandQueue *);

bool LoadCommands(const int argc, const char ** argv, CommandQueue * cmd);

bool WriteOutputFile( const float *value_table,
                       const size_t table_length,
                       const int output_format,
                       const char *output_filename);

bool RecordProgress( const float *first_table,
                      const float *second_table,
                      const size_t table_length,
                      const char *progress_file_name);

bool LoadProgress( float *first_table,
                   float *second_table,
                   const size_t table_length,
                   const char *progress_file_name);

/* void PrintVerboseInfo (); */


#endif   /* ----- #ifndef SUPPORT_H_  ----- */

/* =============================================================================
 *                         end of file support.h
 * =============================================================================
 */
