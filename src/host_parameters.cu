/*
 * =============================================================================
 *
 *       Filename:  host_parameters.cc
 *
 *    Description:  The implementation of HostParameters
 *
 *        Created:  Tue Jul 28 14:58:27 2015
 *       Modified:  Sat Aug  1 13:38:38 2015
 *
 *         Author:  Huang Zonghao
 *          Email:  coding@huangzonghao.com
 *
 * =============================================================================
 */
#include <memory>

#include "host_parameters.h"
#include "device_parameters.h"
/* =============================================================================
 *                  Methods of HostParameters
 * =========================================================================== */

/*
 *------------------------------------------------------------------------------
 *       Class:  HostParameters
 *      Method:  HostParameters
 * Description:  constructor
 *------------------------------------------------------------------------------
 */
HostParameters::HostParameters () {
              T =   0;
              m =   0;
              k =   0;
        maxhold =   0;
              c = 0.0;
              h = 0.0;
          theta = 0.0;
              r = 0.0;
              s = 0.0;
          alpha = 0.0;
         lambda = 0.0;
     min_demand =   0;
     max_demand =   0;
}  /* -----  end of method HostParameters::HostParameters  (constructor)  ----- */

/*
 *------------------------------------------------------------------------------
 *       Class:  HostParameters
 *      Method:  HostParameters
 * Description:  copy constructor
 *------------------------------------------------------------------------------
 */
HostParameters::HostParameters ( const HostParameters &other ) {
    if (this != &other) {
        T          = other.T;
        m          = other.m;
        k          = other.k;
        maxhold    = other.maxhold;
        c          = other.c;
        h          = other.h;
        theta      = other.theta;
        r          = other.r;
        s          = other.s;
        alpha      = other.alpha;
        lambda     = other.lambda;
        min_demand = other.min_demand;
        max_demand = other.max_demand;

        if (max_demand - min_demand != 0 ) {
            demand_distribution = malloc((max_demand - min_demand) *\
                                            sizeof(float));
            for (int i = 0; i < max_demand - min_demand; ++i){
                demand_distribution[i] = other.demand_distribution[i];
            }
        }
}  /* -----  end of method HostParameters::HostParameters  (copy constructor)  ----- */


/*
 *------------------------------------------------------------------------------
 *       Class:  HostParameters
 *      Method:  operator =
 * Description:  assignment operator
 *------------------------------------------------------------------------------
 */
HostParameters&
HostParameters::operator = ( const HostParameters &other ) {
    if ( this != &other ) {
        T       = other.T;
        m       = other.m;
        k       = other.k;
        maxhold = other.maxhold;
        c       = other.c;
        h       = other.h;
        theta   = other.theta;
        r       = other.r;
        s       = other.s;
        alpha   = other.alpha;
        lambda  = other.lambda;

        if (demand_distribution != NULL ||
            (other.max_demand - other.min_demand\
             != max_demand - min_demand)){
            free(demand_distribution);
        }
        min_demand = other.min_demand;
        max_demand = other.max_demand;

        if (max_demand - min_demand != 0 ) {
            demand_distribution  = malloc((max_demand - min_demand) *\
                                            sizeof(float));
            if ( demand_distribution==NULL ) {
                fprintf(stderr, "\ndynamic memory allocation failed\n");
                exit(EXIT_FAILURE);
            }

            for (int i = 0; i < max_demand - min_demand; ++i){
                demand_distribution[i] = other.demand_distribution[i];
            }
        }
    }
    return *this;
}

/* reload to get the values from cuda device directly */
HostParameters&
HostParameters::operator = ( const DeviceParameters &device ) {
    if (device.IsComplete()){
        read_from_device(&T,       device.T,       1);
        read_from_device(&m,       device.m,       1);
        read_from_device(&k,       device.k,       1);
        read_from_device(&maxhold, device.maxhold, 1);
        read_from_device(&c,       device.c,       1);
        read_from_device(&h,       device.h,       1);
        read_from_device(&theta,   device.theta,   1);
        read_from_device(&r,       device.r,       1);
        read_from_device(&s,       device.s,       1);
        read_from_device(&alpha,   device.alpha,   1);
        read_from_device(&lambda,  device.lambda,  1);
    }
    return *this;
}  /* -----  end of method HostParameters::operator =  (assignment operator)  ----- */

/*
 *------------------------------------------------------------------------------
 *       Class:  HostParameters
 *      Method:  set_value
 * Description:  An interface to set the value of the HostParameters
 *------------------------------------------------------------------------------
 */
bool HostParameters::set_value ( const char * var, float value ) {
    switch ( var ) {
        case "T":
            T = (size_t) value;
            break;
        case "m":
            m = (size_t) value;
            break;
        case "maxhold":
            maxhold = (size_t) value;
            break;
        case "c":
            c = value;
            break;
        case "h":
            h = value;
            break;
        case "theta":
            theta = value;
            break;
        case "r":
            r = value;
            break;
        case "s":
            s = value;
            break;
        case "alpha":
            alpha = value;
            break;
        case "lambda":
            lambda = value;
            break;
        case "max_demand":
            max_demand = value;
            break;
        case "min_demand":
            min_demand = value;
            break;
        default:
            return false;
        break;
    }
        return true;
}       /* -----  end of method HostParameters::set_value  ----- */

/*
 *------------------------------------------------------------------------------
 *       Class:  HostParameters
 *      Method:  get_value
 * Description:  return the variables' value stored in HostParameters
 *                 and note all the values are returned in float
 *------------------------------------------------------------------------------
 */
float HostParameters::get_value (const char * var) {
    switch ( var ) {
        case "T":
            return (float)T;
        case "m":
            return (float)m;
        case "maxhold":
            return (float)maxhold;
        case "c":
            return c;
        case "h":
            return h;
        case "theta":
            return theta;
        case "r":
            return r;
        case "s":
            return s;
        case "alpha":
            return alpha;
        case "lambda":
            return lambda;
        case "max_demand":
            return (float)max_demand;
        case "min_demand":
            return (float)min_demand;
        default:
            printf("Cannot find the variable\n");
            return 0;
    }
}       /* -----  end of method HostParameters::get_value  ----- */


/*
 *------------------------------------------------------------------------------
 *       Class:  HostParameters
 *      Method:  print_params
 * Description:  print out the parameters stored in HostParameters
 *------------------------------------------------------------------------------
 */
void HostParameters::print_params () {
    printf("===========================\n"\
            "The parameters read are : \e[38;5;166m]\n"\
            "\e[1;33mm : \e[38;5;166m%d\n"\
            "\e[1;33mk : \e[38;5;166m%d\n"\
            "\e[1;33mT : \e[38;5;166m%d\n"\
            "\e[1;33mh : \e[38;5;166m%f\n"\
            "\e[1;33mr : \e[38;5;166m%f\n"\
            "\e[1;33mc : \e[38;5;166m%f\n"\
            "\e[1;33mtheta : \e[38;5;166m%f\n"\
            "\e[1;33ms : \e[38;5;166m%f\n"\
            "\e[1;33malpha : \e[38;5;166m%f\n"\
            "\e[1;33mlambda : \e[38;5;166m%f\n"\
            "\e[1;33mmaxhold : \e[38;5;166m%d\n"\
            "\e[1;33mmax_demand : \e[38;5;166m%d\n"\
            "\e[1;33mmin_demand : \e[38;5;166m%d\n"\
            "\e[m===========================\n",
            m, k, T, h, r, c, theta, s, alpha, lambda, maxhold, max_demand,\
            min_demand);
            return ;
}       /* -----  end of method HostParameters::print_params  ----- */


/* =============================================================================
 *                         end of file host_parameters.cc
 * =============================================================================
 */
