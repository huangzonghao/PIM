/*
 * =============================================================================
 *
 *       Filename:  process_record.h
 *
 *    Description:  This file contains the definition of class ProgressRecord
 *
 *        Created:  Tue Jul 28 17:14:03 2015
 *       Modified:  Tue Jul 28 17:17:12 2015
 *
 *         Author:  Huang Zonghao
 *          Email:  coding@huangzonghao.com
 *
 * =============================================================================
 */
#ifndef PROCESS_RECORD_H_
#define PROCESS_RECORD_H_
/*
 * =============================================================================
 *        Class:  ProgressRecord
 *  Description:  Contains the descritpiton of the progress of the current
 *                  task. Mainly for keeping the log and susbending and
 *                  restarting the task.
 * =============================================================================
 */
class ProgressRecord {
  public:
    /* =========================   LIFECYCLE   =============================== */

    /* constructor */
    ProgressRecord ();
    /* copy constructor */
    ProgressRecord ( const ProgressRecord &other );
    /* destructor */
    ~ProgressRecord ();

    /* =========================   ACCESSORS   =============================== */

    /* =========================   MUTATORS    =============================== */

    /* =========================   OPERATORS   =============================== */

    /* assignment operator */
    ProgressRecord& operator = ( const ProgressRecord &other );

  protected:
    /* ========================  DATA MEMBERS  =============================== */

  private:
    /* ========================  DATA MEMBERS  =============================== */

}; /* -----  end of class ProgressRecord  ----- */







#endif   /* ----- #ifndef PROCESS_RECORD_H_  ----- */

/* =============================================================================
 *                         end of file process_record.h
 * =============================================================================
 */
