/*
 * =============================================================================
 *
 *       Filename:  support.h
 *
 *    Description:  This is the header file of support.h
 *
 *        Created:  Wed Jul 22 18:38:32 2015
 *       Modified:  Thu Aug 27 13:05:37 2015
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

void PrintUsage ();

void InterruptHandler ( int s );

bool LoadParameters ( CommandQueue * );

bool LoadCommands (const int argc, const char ** argv, CommandQueue * cmd);

bool WriteOutputFile ( const float *value_table,
                       const std::string &output_format,
                       const std::string &output_filename );

bool RecordProgress ( const float *current_value_table,
                      const float *prev_value_table,
                      const std::string &record_label );

bool LoadProgress ( const std::string &record_filename,
                    const float *current_value_table,
                    const float *prev_value_table );

void PrintVerboseInfo ();


#endif   /* ----- #ifndef SUPPORT_H_  ----- */

/* =============================================================================
 *                         end of file support.h
 * =============================================================================
 */
