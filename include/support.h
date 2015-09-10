/*
 * =============================================================================
 *
 *       Filename:  support.h
 *
 *    Description:  This is the header file of support.h
 *
 *        Created:  Wed Jul 22 18:38:32 2015
 *       Modified:  Thu 10 Sep 2015 02:55:37 PM HKT
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

/* #ifdef _DEFINE_CONST_VARS_IN_HEADER_ */
/* the supported output file formats */
static const int num_file_format_options = 4;
static const char *file_format_options[] = { "csv",
                                             "nature",
                                             "json",
                                             "xml" };
/* #endif */
/* #####   EXPORTED FUNCTION DECLARATIONS   ################################## */

void PrintUsage();

void InterruptHandler(int s);

bool LoadParameters(CommandQueue *);

bool LoadCommands(int argc, char **argv, CommandQueue *cmd);

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
