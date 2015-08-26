/*
 * =============================================================================
 *
 *       Filename:  demand_distribution.h
 *
 *    Description:  The definition of the DemandDistribution structure
 *
 *        Created:  Wed Aug 26 12:15:04 2015
 *       Modified:  Wed Aug 26 13:02:05 2015
 *
 *         Author:  Huang Zonghao
 *          Email:  coding@huangzonghao.com
 *
 * =============================================================================
 */
#ifndef DEMAND_DISTRIBUTION_H_
#define DEMAND_DISTRIBUTION_H_

#define MAX_DISTRIBUTION_LENGTH 60
#include <stdlib.h>

/*
 * ===  SRUCT  =================================================================
 *         Name:  DemandDistribution
 *  Description:  each instance of this structure stores the distribution for
 *                  one period. So to store the distribution for mutiple periods
 *                  use mutiple instances
 * =============================================================================
 */
struct DemandDistribution {
    size_t min_demand;
    size_t max_demand;
    float table[MAX_DISTRIBUTION_LENGTH];
};       /* ----------  end of struct DemandDistribution  ---------- */

/*
 * ===  FUNCTION  ==============================================================
 *         Name:  CopyDemandDistribution
 *  Description:  copy the structure of DemandDistribution
 *       @param:  source, target
 *      @return:  bool
 * =============================================================================
 */
inline bool CopyDemandDistribution(DemandDistribution &source, DemandDistribution &target){
    target.min_demand = source.min_demand;
    target.max_demand = source.max_demand;
    for (int i = 0; i < (int)(target.max_demand - target.min_demand) + 1; ++i){
        target.table[i] = source.table[i];
    }
    return true;
}       /* -----  end of function CopyDemandDistribution  ----- */
#endif   /* ----- #ifndef DEMAND_DISTRIBUTION_H_  ----- */
/* =============================================================================
 *                         end of file demand_distribution.h
 * =============================================================================
 */
