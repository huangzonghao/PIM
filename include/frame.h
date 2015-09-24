/*
 * =============================================================================
 *
 *       Filename:  frame.h
 *
 *    Description:   The header file of frame.cc
 *
 *        Created:  Fri Aug  7 18:03:09 2015
 *       Modified:  Thu 24 Sep 2015 09:01:13 AM HKT
 *
 *         Author:  Huang Zonghao
 *          Email:  coding@huangzonghao.com
 *
 * =============================================================================
 */
#ifndef FRAME_H_
#define FRAME_H_
#include <vector>
class CommandQueue;
class SystemInfo;

bool LetsRock( CommandQueue * cmd,
               SystemInfo * sysinfo,
               std::vector<float*> &host_value_tables );

#endif   /* ----- #ifndef FRAME_H_  ----- */
/* =============================================================================
 *                         end of file frame.h
 * =============================================================================
 */
