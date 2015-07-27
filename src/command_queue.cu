/*
 * =============================================================================
 *
 *       Filename:  command_queue.cu
 *
 *    Description:  This file contains the definition of methods of the classes
 *                    defined in comand_queue.h
 *
 *        Created:  Fri Jul 24 13:52:37 2015
 *       Modified:  Tue Jul 28 02:30:37 2015
 *
 *         Author:  Huang Zonghao
 *          Email:  coding@huangzonghao.com
 *
 * =============================================================================
 */
#include <cuda.h>
#include <cuda_runtime.h>

#include "command_queue.h"

/* =============================================================================
 *                  Methods of CommandQueue
 * =========================================================================== */

/*
 *------------------------------------------------------------------------------
 *       Class:  CommandQueue
 *      Method:  CommandQueue
 * Description:  copy constructor
 *------------------------------------------------------------------------------
 */
CommandQueue::CommandQueue ( const CommandQueue &other ) {
    if (this != other){
        h = other.h;
        if (other.d.IsComplete()){
            d = other.d;
        }
    }
}  /* -----  end of method CommandQueue::CommandQueue  (copy constructor)  ----- */


/*
 *------------------------------------------------------------------------------
 *       Class:  CommandQueue
 *      Method:  operator =
 * Description:  assignment operator
 *------------------------------------------------------------------------------
 */
CommandQueue&
CommandQueue::operator = ( const CommandQueue &other ) {
    if ( this != &other ) {
        h = other.h;
        if (other.d.IsComplete()){
            d = other.d;
        }
    }
    return *this;
}  /* -----  end of method CommandQueue::operator =  (assignment operator)  ----- */

/*
 *------------------------------------------------------------------------------
 *       Class:  CommandQueue
 *      Method:  get_host_params
 * Description:  return the pointer to the HostParameters
 *------------------------------------------------------------------------------
 */
<+FUNC_TYPE+> CommandQueue::get_host_params ( <+argument list+> ) {
    <+FUNCTION+>
    return ;
}       /* -----  end of method CommandQueue::get_host_params  ----- */

/*
 *------------------------------------------------------------------------------
 *       Class:  CommandQueue
 *      Method:  get_device_params
 * Description:  return th e pointer to the DeviceParameters
 *------------------------------------------------------------------------------
 */
<+FUNC_TYPE+> CommandQueue::get_device_params ( <+argument list+> ) {
    <+FUNCTION+>
    return ;
}       /* -----  end of method CommandQueue::get_device_params  ----- */

/*
 *------------------------------------------------------------------------------
 *       Class:  CommandQueue
 *      Method:  load_host_params
 * Description:  load the specific value to HostParameters
 *------------------------------------------------------------------------------
 */
<+FUNC_TYPE+> CommandQueue::load_host_params ( <+argument list+> ) {
    <+FUNCTION+>
    return ;
}       /* -----  end of method CommandQueue::load_host_params  ----- */

/*
 *------------------------------------------------------------------------------
 *       Class:  CommandQueue
 *      Method:  update_device_params
 * Description:  passt the params stored in the HostParameters to
 *                 DeviceParameters
 *------------------------------------------------------------------------------
 */
<+FUNC_TYPE+> CommandQueue::update_device_params ( <+argument list+> ) {
    <+FUNCTION+>
    return ;
}       /* -----  end of method CommandQueue::update_device_params  ----- */

/*
 *------------------------------------------------------------------------------
 *       Class:  CommandQueue
 *      Method:  retrieve_device_params
 * Description:  copy the params stored in the DeviceParameters back to
 *                 HostParameters
 *------------------------------------------------------------------------------
 */
<+FUNC_TYPE+> CommandQueue::retrieve_device_params ( <+argument list+> ) {
    <+FUNCTION+>
    return ;
}       /* -----  end of method CommandQueue::retrieve_device_params  ----- */




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
     min_demand = 0;
     max_demand = 0;
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



/* =============================================================================
 *                      Methodss of DeviceParameters
 * =========================================================================== */


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
        checkCudaErrors(cudaMalloc(&demand_distribution, (other.max_demand - other.min_demand) * sizeof(float)));
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
DeviceParameters::operator = ( const HostParameters &other ) {
    /* first check if the object is capable to hold a device copy */
    if (is_owner){
        cerr << "You cannot assign to a owner of a device copy, use the update function" << endl;
        exit(EXIT_FAILURE);
    }

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

    return *this;
}
/* -----  end of method DeviceParameters::operator =  (assignment operator)  ----- */



/*
 *------------------------------------------------------------------------------
 *       Class:  DeviceParameters
 *      Method:  IsComplete
 * Description:  validation of the completeness of the object -- whether all
 *                 the pointers exist
 *------------------------------------------------------------------------------
 */
bool DeviceParameters::IsComplete () {
    if ( is_target_set ==    0 ||
         T             == NULL ||
         m             == NULL ||
         k             == NULL ||
         maxhold       == NULL ||
         c             == NULL ||
         h             == NULL ||
         theta         == NULL ||
         r             == NULL ||
         s             == NULL ||
         alpha         == NULL ||
         lambda        ==   NULL)
            return false;
    else return true;
}       /* -----  end of method DeviceParameters::IsComplete  ----- */


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
        case 'T':
            pass_to_device(&(size_t)value, T, 1);
            break;

        case m:
            pass_to_device(&(size_t)value, m, 1);
            break;

        case maxhold:
            pass_to_device(&(size_t)value, maxhold, 1);
            break;
        case c:
            pass_to_device(&(float)value, c, 1);
            break;
        case h:
            pass_to_device(&(float)value, h, 1);
            break;
        case theta:
            pass_to_device(&(float)value, theta, 1);
            break;
        case r:
            pass_to_device(&(float)value, r, 1);
            break;
        case s:
            pass_to_device(&(float)value, s, 1);
            break;
        case alpha:
            pass_to_device(&(float)value, alpha, 1);
            break;

        case lambda:
            pass_to_device(&(float)value, lambda, 1);
            break;

        default:
            return false;
        break;
}            /* -----  end switch  ----- */
        return true;
}       /* -----  end of method DeviceParameters::set_value  ----- */





/* =============================================================================
 *                         end of file command_queue.cu
 * =============================================================================
 */
