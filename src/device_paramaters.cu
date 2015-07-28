/*
 * =============================================================================
 *
 *       Filename:  device_paramaters.cu
 *
 *    Description:  The implementation of DeviceParameters
 *
 *        Created:  Tue Jul 28 14:59:58 2015
 *       Modified:  Tue Jul 28 20:37:14 2015
 *
 *         Author:  Huang Zonghao
 *          Email:  coding@huangzonghao.com
 *
 * =============================================================================
 */
#include <cuda.h>
#include "../thirdparty/nvidia/helper_cuda.h"
#include "host_parameters.h"
#include "device_parameters.h"
/*
 *------------------------------------------------------------------------------
 *       Class:  DeviceParameters
 *      Method:  DeviceParameters
 * Description:  constructor, and will automatically allocate the variable
 *                 spaces in device
 *------------------------------------------------------------------------------
 */

/* :REMARKS:Mon Jul 27 03:52:39 2015:huangzonghao:
 *  we should do mem allocation at the declaration of a deviceparams class
 *  we just want to keep one copy of the device parameters no matter how
 *  many pointers there pointing to that copy.
 *  so only do memalloc when necessary
 */
DeviceParameters::DeviceParameters (const &HostParameters other) {
    checkCudaErrors(cudaMalloc(&T, sizeof(size_t)));
    checkCudaErrors(cudaMalloc(&m, sizeof(size_t)));
    checkCudaErrors(cudaMalloc(&k, sizeof(size_t)));
    checkCudaErrors(cudaMalloc(&maxhold, sizeof(size_t)));
    checkCudaErrors(cudaMalloc(&c, sizeof(float)));
    checkCudaErrors(cudaMalloc(&h, sizeof(float)));
    checkCudaErrors(cudaMalloc(&theta, sizeof(float)));
    checkCudaErrors(cudaMalloc(&r, sizeof(float)));
    checkCudaErrors(cudaMalloc(&s, sizeof(float)));
    checkCudaErrors(cudaMalloc(&alpha, sizeof(float)));
    checkCudaErrors(cudaMalloc(&lambda, sizeof(float)));
    checkCudaErrors(cudaMalloc(&min_demand, sizeof(size_t)));
    checkCudaErrors(cudaMalloc(&max_demand, sizeof(size_t)));

    if ( other.max_demand - other.min_demand != 0 ){
        checkCudaErrors(cudaMalloc(&demand_distribution,\
                    (other.max_demand - other.min_demand) * sizeof(float)));
    }
    pass_to_device(&other.T,          T,          1);
    pass_to_device(&other.m,          m,          1);
    pass_to_device(&other.k,          k,          1);
    pass_to_device(&other.maxhold,    maxhold,    1);
    pass_to_device(&other.c,          c,          1);
    pass_to_device(&other.h,          h,          1);
    pass_to_device(&other.theta,      theta,      1);
    pass_to_device(&other.r,          r,          1);
    pass_to_device(&other.s,          s,          1);
    pass_to_device(&other.alpha,      alpha,      1);
    pass_to_device(&other.lambda,     lambda,     1);
    pass_to_device(&other.max_demand, maxhold,    1);
    pass_to_device(&other.min_demand, min_demand, 1);

    pass_to_device(&other.demand_distribution, demand_distribution,\
                    other.max_demand - other.min_demand);

    is_target_set = 1;
    is_owner = 1;
    /* :REMARKS:Mon Jul 27 03:49:49 2015:huangzonghao:
     *  still cannot allocate the device table here
     */
}  /* -----  end of method DeviceParameters::DeviceParameters  (constructor)  ----- */

/*
 *------------------------------------------------------------------------------
 *       Class:  DeviceParameters
 *      Method:  DeviceParameters
 * Description:  copy constructor
 *------------------------------------------------------------------------------
 */
DeviceParameters::DeviceParameters ( const DeviceParameters &other ) {
    if ( this != &other ) {
        T                   = other.T;
        m                   = other.m;
        k                   = other.k;
        maxhold             = other.maxhold;
        c                   = other.c;
        h                   = other.h;
        theta               = other.theta;
        r                   = other.r;
        s                   = other.s;
        alpha               = other.alpha;
        lambda              = other.lambda;
        max_demand          = other.max_demand;
        min_demand          = other.min_demand;
        demand_distribution = other.demand_distribution;
    }
    is_target_set = 1;
    is_owner = 0;
}  /* -----  end of method DeviceParameters::DeviceParameters  (copy constructor)  ----- */

/*
 *------------------------------------------------------------------------------
 *       Class:  DeviceParameters
 *      Method:  ~DeviceParameters
 * Description:  destructor
 *------------------------------------------------------------------------------
 */
/* :TODO:Mon Jul 27 04:19:06 2015:huangzonghao:
 *  think about the race thing when doing destruction!!!!
 */
DeviceParameters::~DeviceParameters () {
    if (is_owner && is_target_set){
        checkCudaErrors(cudaFree(T));
        checkCudaErrors(cudaFree(m));
        checkCudaErrors(cudaFree(k));
        checkCudaErrors(cudaFree(maxhold));
        checkCudaErrors(cudaFree(c));
        checkCudaErrors(cudaFree(h));
        checkCudaErrors(cudaFree(theta));
        checkCudaErrors(cudaFree(r));
        checkCudaErrors(cudaFree(s));
        checkCudaErrors(cudaFree(alpha));
        checkCudaErrors(cudaFree(lambda));
        checkCudaErrors(cudaFree(min_demand));
        checkCudaErrors(cudaFree(max_demand));

        if (demand_distribution != NULL){
            checkCudaErrors(cudaFree(demand_distribution));
        }
    }
}  /* -----  end of method DeviceParameters::~DeviceParameters  (destructor)  ----- */

/*
 *------------------------------------------------------------------------------
 *       Class:  DeviceParameters
 *      Method:  operator =
 * Description:  assignment operator
 *------------------------------------------------------------------------------
 */
DeviceParameters&
DeviceParameters::operator = ( const DeviceParameters &other ) {
    if ( this != &other ) {
        T                   = other.T;
        m                   = other.m;
        k                   = other.k;
        maxhold             = other.maxhold;
        c                   = other.c;
        h                   = other.h;
        theta               = other.theta;
        r                   = other.r;
        s                   = other.s;
        alpha               = other.alpha;
        lambda              = other.lambda;
        max_demand          = other.max_demand;
        min_demand          = other.min_demand;
        demand_distribution = other.demand_distribution;
    }
    is_target_set = 1;
    is_owner = 0;
    return *this;
}


/* :REMARKS:Mon Jul 27 01:47:27 2015:huangzonghao:
 * Now with the following function, loading the parameters to the device
 * becomes fairly easy, we just need to load to the host first then let the
 * device equal to the host
 */
DeviceParameters&
DeviceParameters::operator = ( const HostParameters &hp ) {
    /* first check if the object is capable to hold a device copy */
    if (is_owner){
        printf( "You cannot assign to a owner of"\
                " a device copy, use the update function\n");
        exit(EXIT_FAILURE);
    }

    checkCudaErrors(cudaMalloc(&T,          sizeof(size_t)));
    checkCudaErrors(cudaMalloc(&m,          sizeof(size_t)));
    checkCudaErrors(cudaMalloc(&k,          sizeof(size_t)));
    checkCudaErrors(cudaMalloc(&maxhold,    sizeof(size_t)));
    checkCudaErrors(cudaMalloc(&c,           sizeof(float)));
    checkCudaErrors(cudaMalloc(&h,           sizeof(float)));
    checkCudaErrors(cudaMalloc(&theta,       sizeof(float)));
    checkCudaErrors(cudaMalloc(&r,           sizeof(float)));
    checkCudaErrors(cudaMalloc(&s,           sizeof(float)));
    checkCudaErrors(cudaMalloc(&alpha,       sizeof(float)));
    checkCudaErrors(cudaMalloc(&lambda,      sizeof(float)));
    checkCudaErrors(cudaMalloc(&min_demand, sizeof(size_t)));
    checkCudaErrors(cudaMalloc(&max_demand, sizeof(size_t)));

    if ( hp.max_demand - hp.min_demand != 0 ){
        checkCudaErrors(cudaMalloc(&demand_distribution,\
                    (hp.max_demand - hp.min_demand) * sizeof(float)));
    }
    pass_to_device((size_t * )hp.get_ptr("T"),          T,           1);
    pass_to_device((size_t * )hp.get_ptr("m"),          m,           1);
    pass_to_device((size_t * )hp.get_ptr("k"),          k,           1);
    pass_to_device((size_t * )hp.get_ptr("maxhold"),    maxhold,     1);
    pass_to_device((float  * )hp.get_ptr("c"),           c,          1);
    pass_to_device((float  * )hp.get_ptr("h"),           h,          1);
    pass_to_device((float  * )hp.get_ptr("theta"),       theta,      1);
    pass_to_device((float  * )hp.get_ptr("r"),           r,          1);
    pass_to_device((float  * )hp.get_ptr("s"),           s,          1);
    pass_to_device((float  * )hp.get_ptr("alpha"),       alpha,      1);
    pass_to_device((float  * )hp.get_ptr("lambda"),      lambda,     1);
    pass_to_device((size_t * )hp.get_ptr("max_demand"), maxhold,     1);
    pass_to_device((size_t * )hp.get_ptr("min_demand"), min_demand,  1);

    pass_to_device(&hp.demand_distribution, demand_distribution,\
                    hp.max_demand - hp.min_demand);
    is_target_set = 1;
    is_owner = 1;

    return *this;
}
/* -----  end of method DeviceParameters::operator =  (assignment operator)  ----- */



/*
 *------------------------------------------------------------------------------
 *       Class:  DeviceParameters
 *      Method:  is_complete
 * Description:  validation of the completeness of the object -- whether all
 *                 the pointers exist
 *------------------------------------------------------------------------------
 */
bool DeviceParameters::is_complete () {
    if ( is_target_set_ == 0    ||
         T              == NULL ||
         m              == NULL ||
         k              == NULL ||
         maxhold        == NULL ||
         c              == NULL ||
         h              == NULL ||
         theta          == NULL ||
         r              == NULL ||
         s              == NULL ||
         alpha          == NULL ||
         lambda         == NULL)
            return false;
    else return true;
}       /* -----  end of method DeviceParameters::is_complete  ----- */
/*
 *------------------------------------------------------------------------------
 *       Class:  DeviceParameters
 *      Method:  is_owner
 * Description:  check if the this struct is the owner of some device data
 *                 copy
 *------------------------------------------------------------------------------
 */
bool DeviceParameters::is_owner () {
    if (is_owner_)
        return true;
    else return false ;
}       /* -----  end of method DeviceParameters::is_owner  ----- */

/*
 *------------------------------------------------------------------------------
 *       Class:  DeviceParameters
 *      Method:  is_linked
 * Description:  check if the current struct is linked to some device data copy
 *------------------------------------------------------------------------------
 */
bool DeviceParameters::is_linked () {
    if (is_target_set_)
        return true;
    else return false;
}       /* -----  end of method DeviceParameters::is_linked  ----- */


/*
 *------------------------------------------------------------------------------
 *       Class:  DeviceParameters
 *      Method:  set_value
 * Description:  pass the desired value to device, but should be rarely used
 *                 since    the device variables should always be synchorized
 *                 with the host copy
 *------------------------------------------------------------------------------
 */
bool DeviceParameters::set_value ( char * var, float value ) {
    switch ( var ) {
        case "c":
            pass_to_device(&value, c, 1);
            break;
        case "h":
            pass_to_device(&value, h, 1);
            break;
        case "theta":
            pass_to_device(&value, theta, 1);
            break;
        case "r":
            pass_to_device(&value, r, 1);
            break;
        case "s":
            pass_to_device(&value, s, 1);
            break;
        case "alpha":
            pass_to_device(&value, alpha, 1);
            break;

        case "lambda":
            pass_to_device(&value, lambda, 1);
            break;

        default:
            return false;
            break;
    }            /* -----  end switch  ----- */
    return true;
}
bool DeviceParameters::set_value ( char * var, size_t value ) {
    switch ( var ) {
        case "T":
            pass_to_device(&value, T, 1);
            break;

        case "m":
            pass_to_device(&value, m, 1);
            break;

        case "maxhold":
            pass_to_device(&value, maxhold, 1);
            break;

        default:
            return false;
            break;
    }            /* -----  end switch  ----- */
    return true;
}       /* -----  end of method DeviceParameters::set_value  ----- */



/* =============================================================================
 *                         end of file device_paramaters.cu
 * =============================================================================
 */
