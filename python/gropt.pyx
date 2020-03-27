import numpy as np
cimport numpy as np
import time
cimport cython

global N_ddebug
N_ddebug = 100


cdef extern from "../src/optimize_kernel.c":
    void _run_kernel_diff_fixeddt "run_kernel_diff_fixeddt"(double **G_out, int *N_out, double **ddebug, int verbose, 
                                                        double dt0, double gmax, double smax, double TE, 
                                                        int N_moments, double *moments_params, double PNS_thresh, 
                                                        double T_readout, double T_90, double T_180, int diffmode, double dt_out,
                                                        int N_eddy, double *eddy_params, double search_bval, double slew_reg, int Naxis)

    void _run_kernel_diff_fixedN "run_kernel_diff_fixedN"(double **G_out, int *N_out, double **ddebug, int verbose, 
                                                        int N0, double gmax, double smax, double TE, 
                                                        int N_moments, double *moments_params, double PNS_thresh, 
                                                        double T_readout, double T_90, double T_180, int diffmode, double dt_out,
                                                        int N_eddy, double *eddy_params, double search_bval, double slew_reg)

    void _run_kernel_diff_fixedN_Gin "run_kernel_diff_fixedN_Gin"(double **G_out, int *N_out, double **ddebug, int verbose, 
                                                                double *G_in, int N0, double gmax, double smax, double TE, 
                                                                int N_moments, double *moments_params, double PNS_thresh, 
                                                                double T_readout, double T_90, double T_180, int diffmode,  double dt_out,
                                                                int N_eddy, double *eddy_params, double search_bval, double slew_reg)

    void _run_kernel_diff_fixeddt_fixG "run_kernel_diff_fixeddt_fixG"(double **G_out, int *N_out, double **ddebug, int verbose,
                                                                    double dt0, double gmax, double smax, double TE,
                                                                    int N_moments, double *moments_params, double PNS_thresh, 
                                                                    double T_readout, double T_90, double T_180, int diffmode, double dt_out,
                                                                    int N_eddy, double *eddy_params, double search_bval,
                                                                    int N_gfix, double *gfix, double slew_reg)


def array_prep(A, dtype, linear=True):
    if not A.flags['C_CONTIGUOUS']:
        A = np.ascontiguousarray(A)
    
    A = A.astype(dtype, order='C', copy=False)
    
    if linear:
        A = A.ravel()

    return A


@cython.boundscheck(False) 
@cython.wraparound(False)
def gropt(params, verbose=0):
    
    #--------------
    # Read params with bunches of error checks
    # This is probably excessive vs. just trying to open these fields, but whatevs
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
        TE = params['TE']
    else:
        print('ERROR: params does not contain key "TE"')
        return

    if diffmode > 0:
        if 'T_90' in params:
            T_90 = params['T_90']
        else:
            print('ERROR: params does not contain key "T_90"')
            return
        
        if 'T_180' in params:
            T_180 = params['T_180']
        else:
            print('ERROR: params does not contain key "T_180"')
            return

        if 'T_readout' in params:
            T_readout = params['T_readout']
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
    if N0 > 0:
        _run_kernel_diff_fixedN(&G_out, &N_out, &ddebug, verbose, N0, gmax, smax, TE, N_moments, &moment_params_view[0], 
                                pns_thresh, T_readout, T_90, T_180, diffmode, dt_out, N_eddy, &eddy_params_view[0], -1.0, slew_reg)
    elif dt > 0:
        if N_gfix > 0:
            _run_kernel_diff_fixeddt_fixG(&G_out, &N_out, &ddebug, verbose, dt, gmax, smax, TE, N_moments, &moment_params_view[0], 
                                                                    pns_thresh, T_readout, T_90, T_180, diffmode, dt_out, N_eddy, &eddy_params_view[0], -1.0,
                                                                    N_gfix, &gfix_view[0], slew_reg)
        else:
            _run_kernel_diff_fixeddt(&G_out, &N_out, &ddebug, verbose, dt, gmax, smax, TE, N_moments, &moment_params_view[0], 
                                    pns_thresh, T_readout, T_90, T_180, diffmode, dt_out, N_eddy, &eddy_params_view[0], -1.0, slew_reg, Naxis)
    stop_t = time.time_ns()

    run_time = (stop_t-start_t) / (10 ** 9)

    G_return = np.empty(N_out)
    for i in range(N_out):
        G_return[i] = G_out[i]

    debug_out = np.empty(N_ddebug)
    for i in range(N_ddebug):
        debug_out[i] = ddebug[i]

    G_return = np.reshape(G_return, (Naxis,-1))

    debug_out[15] = run_time

    return G_return, debug_out


def run_diffkernel_fixN(gmax, smax, MMT, TE, T_readout, T_90, T_180, diffmode, N0 = 64, dt_out = -1.0, 
                        eddy = [], pns_thresh = -1.0, verbose = 1):

    m_params = [[0.0, 0.0, -1.0, -1.0, 0.0, 1.0e-3]]
    if MMT > 0:
        m_params.append([0.0, 1.0, -1.0, -1.0, 0.0, 1.0e-3])
    if MMT > 1:
        m_params.append([0.0, 2.0, -1.0, -1.0, 0.0, 1.0e-3])
    m_params = np.array(m_params).flatten()
    eddy = np.array(eddy).flatten() 

    N_moments = m_params.size//6
    m_params = np.ascontiguousarray(np.ravel(m_params), np.float64)
    cdef np.ndarray[np.float64_t, ndim=1, mode="c"] m_params_c = m_params

    cdef double *G_out
    cdef int N_out  
    cdef double *ddebug

    

    N_eddy = eddy.size//4
    if N_eddy > 0:
        eddy_params = np.ascontiguousarray(np.ravel(eddy), np.float64)
    else:
        eddy_params = np.ascontiguousarray(np.ravel(np.zeros(4)), np.float64)
    cdef np.ndarray[np.float64_t, ndim=1, mode="c"] eddy_params_c = eddy_params

    _run_kernel_diff_fixedN(&G_out, &N_out, &ddebug, verbose, N0, gmax, smax, TE, N_moments, &m_params_c[0], pns_thresh, T_readout, T_90, T_180, diffmode, dt_out, N_eddy, &eddy_params_c[0], -1.0, 1.0)

    G_return = np.zeros(N_out)
    for i in range(N_out):
        G_return[i] = G_out[i]

    debug_out = np.zeros(N_ddebug)
    for i in range(N_ddebug):
        debug_out[i] = ddebug[i]

    return G_return, debug_out


def run_diffkernel_fixN_Gin(G_in, gmax, smax, MMT, TE, T_readout, T_90, T_180, diffmode, N0 = 64, dt_out = -1.0, 
                            eddy = [], pns_thresh = -1.0, verbose = 1):

    m_params = [[0.0, 0.0, -1.0, -1.0, 0.0, 1.0e-3]]
    if MMT > 0:
        m_params.append([0.0, 1.0, -1.0, -1.0, 0.0, 1.0e-3])
    if MMT > 1:
        m_params.append([0.0, 2.0, -1.0, -1.0, 0.0, 1.0e-3])
    m_params = np.array(m_params).flatten()
    eddy = np.array(eddy).flatten() 

    N_moments = m_params.size//6
    m_params = np.ascontiguousarray(np.ravel(m_params), np.float64)
    cdef np.ndarray[np.float64_t, ndim=1, mode="c"] m_params_c = m_params

    cdef double *G_out
    cdef int N_out  
    cdef double *ddebug

    cdef np.ndarray[np.float64_t, ndim=1, mode="c"] G_in_c = np.ascontiguousarray(np.ravel(G_in), np.float64)

    N_eddy = eddy.size//4
    if N_eddy > 0:
        eddy_params = np.ascontiguousarray(np.ravel(eddy), np.float64)
    else:
        eddy_params = np.ascontiguousarray(np.ravel(np.zeros(4)), np.float64)
    cdef np.ndarray[np.float64_t, ndim=1, mode="c"] eddy_params_c = eddy_params

    _run_kernel_diff_fixedN_Gin(&G_out, &N_out, &ddebug, verbose, &G_in_c[0], N0, gmax, smax, TE, N_moments, &m_params_c[0], pns_thresh, T_readout, T_90, T_180, diffmode, dt_out, N_eddy, &eddy_params_c[0], -1.0, 1.0)

    G_return = np.zeros(N_out)
    for i in range(N_out):
        G_return[i] = G_out[i]

    debug_out = np.zeros(N_ddebug)
    for i in range(N_ddebug):
        debug_out[i] = ddebug[i]

    return G_return, debug_out


def run_diffkernel_fixdt(gmax, smax, MMT, TE, T_readout, T_90, T_180, diffmode, dt = 0.4e-3, dt_out = -1.0, 
                         eddy = [], pns_thresh = -1.0, verbose = 1):

    m_params = [[0.0, 0.0, -1.0, -1.0, 0.0, 1.0e-3]]
    if MMT > 0:
        m_params.append([0.0, 1.0, -1.0, -1.0, 0.0, 1.0e-3])
    if MMT > 1:
        m_params.append([0.0, 2.0, -1.0, -1.0, 0.0, 1.0e-3])
    m_params = np.array(m_params).flatten() 
    eddy = np.array(eddy).flatten() 

    N_moments = m_params.size//6
    m_params = np.ascontiguousarray(np.ravel(m_params), np.float64)
    cdef np.ndarray[np.float64_t, ndim=1, mode="c"] m_params_c = m_params

    cdef double *G_out
    cdef int N_out  
    cdef double *ddebug

    N_eddy = eddy.size//4
    if N_eddy > 0:
        eddy_params = np.ascontiguousarray(np.ravel(eddy), np.float64)
    else:
        eddy_params = np.ascontiguousarray(np.ravel(np.zeros(4)), np.float64)
    cdef np.ndarray[np.float64_t, ndim=1, mode="c"] eddy_params_c = eddy_params

    _run_kernel_diff_fixeddt(&G_out, &N_out, &ddebug, verbose, dt, gmax, smax, TE, N_moments, &m_params_c[0], pns_thresh, T_readout, T_90, T_180, diffmode, dt_out, N_eddy, &eddy_params_c[0], -1.0, 1.0, 1)

    G_return = np.zeros(N_out)
    for i in range(N_out):
        G_return[i] = G_out[i]

    debug_out = np.zeros(N_ddebug)
    for i in range(N_ddebug):
        debug_out[i] = ddebug[i]

    return G_return, debug_out


def run_kernel_fixdt(gmax, smax, m_params, TE, T_readout, T_90, T_180, diffmode, dt = 0.4e-3, dt_out = -1.0, 
                         eddy = [], pns_thresh = -1.0, verbose = 1):

    m_params = np.array(m_params).flatten() 
    eddy = np.array(eddy).flatten() 

    N_moments = m_params.size//6
    m_params = np.ascontiguousarray(np.ravel(m_params), np.float64)
    cdef np.ndarray[np.float64_t, ndim=1, mode="c"] m_params_c = m_params

    cdef double *G_out
    cdef int N_out  
    cdef double *ddebug

    N_eddy = eddy.size//4
    if N_eddy > 0:
        eddy_params = np.ascontiguousarray(np.ravel(eddy), np.float64)
    else:
        eddy_params = np.ascontiguousarray(np.ravel(np.zeros(4)), np.float64)
    cdef np.ndarray[np.float64_t, ndim=1, mode="c"] eddy_params_c = eddy_params

    _run_kernel_diff_fixeddt(&G_out, &N_out, &ddebug, verbose, dt, gmax, smax, TE, N_moments, &m_params_c[0], pns_thresh, T_readout, T_90, T_180, diffmode, dt_out, N_eddy, &eddy_params_c[0], -1.0, 1.0, 1)

    G_return = np.zeros(N_out) 
    for i in range(N_out):
        G_return[i] = G_out[i]

    debug_out = np.zeros(N_ddebug)
    for i in range(N_ddebug):
        debug_out[i] = ddebug[i]
