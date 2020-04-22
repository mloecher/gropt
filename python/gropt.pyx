import numpy as np
cimport numpy as np
import time
cimport cython

cdef extern from "../src/wrappers.cpp":
    void _python_wrapper_v1 "python_wrapper_v1"(double *params0, double *params1, double **out0, double **out1, double **out2, int **outsize)

    void _gropt_legacy "gropt_legacy"(double **G_out, int *N_out, double **ddebug, int verbose, 
                                        double dt0, double gmax, double smax, double TE, 
                                        int N_moments, double *moments_params, double PNS_thresh, 
                                        double T_readout, double T_90, double T_180, int diffmode, double dt_out,
                                        int N_eddy, double *eddy_params, double search_bval, double slew_reg, int Naxis)

def array_prep(A, dtype, linear=True):
    if not A.flags['C_CONTIGUOUS']:
        A = np.ascontiguousarray(A)
    
    A = A.astype(dtype, order='C', copy=False)
    
    if linear:
        A = A.ravel()

    return A 


@cython.boundscheck(False) 
@cython.wraparound(False)
def gropt2(params0, verbose=0):

    # params0 = np.zeros(100)
    cdef double[::1] params0_view = array_prep(params0, np.float64)

    params1 = np.zeros(100)
    cdef double[::1] params1_view = array_prep(params1, np.float64)

    cdef double *out0
    cdef double *out1
    cdef double *out2
    cdef int *outsize

    _python_wrapper_v1(&params0_view[0], &params1_view[0], &out0, &out1, &out2, &outsize)

    G_return = np.empty(outsize[0])
    for i in range(outsize[0]):
        G_return[i] = out0[i]

    debug_return = np.empty(outsize[1])
    for i in range(outsize[1]):
        debug_return[i] = out1[i]

    # print('Outsize0 =', outsize[0])
    # print(G_return)

    return G_return, debug_return

@cython.boundscheck(False) 
@cython.wraparound(False)
def gropt(params, verbose=0):
    
    #--------------
    # Read params with bunches of error checks
    # This is probably excessive vs. just trying to open these fields
    #--------------

    if 'mode' in params:
        mode = params['mode']
    else:
        print('ERROR: params does not contain key "mode"')
        return

    if mode == 'diff_bval':
        diffmode = 2
    elif mode == 'diff_beta':
        diffmode = 1
    elif mode == 'free':
        diffmode = 0
    else:
        print('ERROR: mode = %s is not valid' % mode)
        return

    if 'gmax' in params:
        gmax = params['gmax']
        if gmax > 1.0:
            gmax /= 1000.0
    else:
        print('ERROR: params does not contain key "gmax"')
        return

    if 'smax' in params:
        smax = params['smax']
    else:
        print('ERROR: params does not contain key "smax"')
        return

    if 'TE' in params:
        TE = params['TE'] * 1.0e-3
    else:
        print('ERROR: params does not contain key "TE"')
        return

    if diffmode > 0:
        if 'T_90' in params:
            T_90 = params['T_90'] * 1.0e-3
        else:
            print('ERROR: params does not contain key "T_90"')
            return
        
        if 'T_180' in params:
            T_180 = params['T_180'] * 1.0e-3
        else:
            print('ERROR: params does not contain key "T_180"')
            return

        if 'T_readout' in params:
            T_readout = params['T_readout'] * 1.0e-3
        else:
            print('ERROR: params does not contain key "T_readout"')
            return

        if 'MMT' in params:
            MMT = params['MMT']
        else:
            print('ERROR: params does not contain key "MMT"')
            return

        if 'tol' in params:
            tolerance = params['tol']
        else:
            tolerance = 1.0e-3
        moment_params = [[0, 0, 0, -1, -1, 0, tolerance]]
        if MMT > 0:
            moment_params.append([0, 1, 0, -1, -1, 0, tolerance])
        if MMT > 1:
            moment_params.append([0, 2, 0, -1, -1, 0, tolerance])
    elif diffmode == 0:
        T_readout = 0.0
        T_90 = 0.0
        T_180 = 0.0
        params['T_readout'] = 0.0
        params['T_90'] = 0.0
        params['T_180'] = 0.0
        if 'moment_params' in params:
            moment_params = params['moment_params']
        else:
            print('ERROR: params does not contain key "moment_params"')
            return
    else:
        print('ERROR: something went wrong in diffmode setup')
        return

    if 'N0' in params:
        N0 = params['N0']
        dt = (TE-T_readout) * 1.0e-3 / N0
    elif 'dt' in params:
        dt = params['dt']
        N0 = -1
    else:
        print('ERROR: params does not contain key "dt" or "N0" (need 1)')
        return

    if 'dt_out' in params:
        dt_out = params['dt_out']
    else:
        dt_out = -1

    if 'eddy_params' in params:
        eddy_params = params['eddy_params']
    else:
        eddy_params = np.array([])

    if 'pns_thresh' in params:
        pns_thresh = params['pns_thresh']
    else:
        pns_thresh = -1.0

    if 'gfix' in params:
        gfix = params['gfix']
    else:
        gfix = np.array([])

    if 'slew_reg' in params:
        slew_reg = params['slew_reg']
    else:
        slew_reg = 1.0
    
    if 'Naxis' in params:
        Naxis = params['Naxis']
    else:
        Naxis = 1

    #--------------
    # Done reading params, now run the C routines
    #--------------

    moment_params = np.array(moment_params)
    eddy_params = np.array(eddy_params)
    gfix = np.array(gfix)
    

    cdef int N_moments = moment_params.shape[0]
    if N_moments == 0: # Needed only if bounds-checking is on for some reason
        moment_params = np.array([0.0])
    cdef double[::1] moment_params_view = array_prep(moment_params, np.float64)

    cdef int N_eddy = eddy_params.shape[0]
    if N_eddy == 0: # Needed only if bounds-checking is on for some reason
        eddy_params = np.array([0.0])
    cdef double[::1] eddy_params_view = array_prep(eddy_params, np.float64)

    cdef int N_gfix = gfix.size
    if N_gfix == 0: # Needed only if bounds-checking is on for some reason
        gfix = np.array([0.0])
    cdef double[::1] gfix_view = array_prep(gfix, np.float64)

    cdef double *G_out
    cdef int N_out  
    cdef double *ddebug

    # print(N_gfix)
    start_t = time.time_ns()

    _gropt_legacy(&G_out, &N_out, &ddebug, verbose, dt, gmax, smax, TE, N_moments, &moment_params_view[0], 
                            pns_thresh, T_readout, T_90, T_180, diffmode, dt_out, N_eddy, &eddy_params_view[0], -1.0, slew_reg, Naxis)

    stop_t = time.time_ns()

    run_time = (stop_t-start_t) / (10 ** 9)

    G_return = np.empty(N_out)
    for i in range(N_out):
        G_return[i] = G_out[i]

    N_ddebug = 100
    debug_out = np.empty(N_ddebug)
    for i in range(N_ddebug):
        debug_out[i] = ddebug[i]

    G_return = np.reshape(G_return, (Naxis,-1))

    debug_out[15] = run_time

    return G_return, debug_out