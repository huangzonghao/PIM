/*
 * =============================================================================
 *
 *       Filename:  system_info.h
 *
 *    Description:  This file contains the denifination of SystemInfo
 *
 *        Created:  Fri Jul 24 01:11:47 2015
 *       Modified:  Sun Aug  9 03:53:58 2015
 *
 *         Author:  Huang Zonghao
 *          Email:  coding@huangzonghao.com
 *
 * =============================================================================
 */


/*
 * =============================================================================
 *        Class:  SystemInfo
 *  Description:  all the system configuration information goes to this class,
 *                  of course mainly the cuda info.
 * =============================================================================
 */

 /* :TODO:Fri Jul 24 01:15:39 2015 01:15:huangzonghao:
  * mainly for the number of cores and some other stuff
  */


class SystemInfo
{
  public:
    /* =========================   LIFECYCLE   =============================== */

    /* constructor */
    SystemInfo ();
    /* copy constructor */
    SystemInfo ( const SystemInfo &other );
    /* destructor */
    ~SystemInfo ();

    /* =========================   ACCESSORS   =============================== */
    int get_value(const char *);

    /* =========================   MUTATORS    =============================== */

    /* =========================   OPERATORS   =============================== */
    void print_sys_info();
    void check_gpu(); // if no gpu is there, exit from inside

  protected:
    /* ========================  DATA MEMBERS  =============================== */

  private:
    /* ========================  DATA MEMBERS  =============================== */
    int num_cuda_devices_;
    int num_cuda_cores_;
    int cuda_core_size_; // number of threads per block

}; /* -----  end of class SystemInfo  ----- */






/* =============================================================================
 *                         end of file system_info.h
 * =============================================================================
 */
