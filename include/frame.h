/*
 * =============================================================================
 *
 *       Filename:  frame.h
 *
 *    Description:   The header file of frame.cc
 *
 *        Created:  Fri Aug  7 18:03:09 2015
 *       Modified:  Fri Aug 28 08:38:51 2015
 *
 *         Author:  Huang Zonghao
 *          Email:  coding@huangzonghao.com
 *
 * =============================================================================
 */
#ifndef FRAME_H_
#define FRAME_H_
class CommandQueue;
class SystemInfo;

bool LetsRock(CommandQueue * cmd, SystemInfo * sysinfo, float* host_value_table);

#endif   /* ----- #ifndef FRAME_H_  ----- */
/* =============================================================================
 *                         end of file frame.h
 * =============================================================================
 */
