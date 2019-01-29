import numpy as np
cimport numpy as np
import time

cdef extern from "../src/optimize_kernel.c":
    void _run_kernel_diff_fixeddt "run_kernel_diff_fixeddt"(double **G_out, int *N_out, double **ddebug, int verbose, 
                                                        double dt0, double gmax, double smax, double TE, 
                                                        int N_moments, double *moments_params, double PNS_thresh, 
                                                        double T_readout, double T_90, double T_180, int diffmode, double dt_out,
                                                        int N_eddy, double *eddy_params)

    void _run_kernel_diff_fixedN "run_kernel_diff_fixedN"(double **G_out, int *N_out, double **ddebug, int verbose, 
                                                        int N0, double gmax, double smax, double TE, 
                                                        int N_moments, double *moments_params, double PNS_thresh, 
                                                        double T_readout, double T_90, double T_180, int diffmode, double dt_out,
                                                        int N_eddy, double *eddy_params)

    void _run_kernel_diff_fixedN_Gin "run_kernel_diff_fixedN_Gin"(double **G_out, int *N_out, double **ddebug, int verbose, 
                                                                double *G_in, int N0, double gmax, double smax, double TE, 
                                                                int N_moments, double *moments_params, double PNS_thresh, 
                                                                double T_readout, double T_90, double T_180, int diffmode,  double dt_out,
                                                                int N_eddy, double *eddy_params)

def run_diffkernel_fixN(gmax, smax, MMT, TE, T_readout, T_90, T_180, diffmode, N0 = 64, dt_out = -1.0, 
                        eddy = [], pns_thresh = -1.0, verbose = 1):

    m_params = [[0.0, 0.0, -1.0, -1.0, 0.0, 1.0e-3]]
    if MMT > 0:
        m_params.append([0.0, 1.0, -1.0, -1.0, 0.0, 1.0e-3])
    if MMT > 1:
        m_params.append([0.0, 2.0, -1.0, -1.0, 0.0, 1.0e-3])
    m_params = np.array(m_params).flatten()

    N_moments = m_params.size//6
    m_params = np.ascontiguousarray(np.ravel(m_params), np.float64)
    cdef np.ndarray[np.float64_t, ndim=1, mode="c"] m_params_c = m_params

    cdef double *G_out
    cdef int N_out  
    cdef double *ddebug

    

    N_eddy = len(eddy)//2
    if N_eddy > 0:
        eddy_params = np.ascontiguousarray(np.ravel(eddy), np.float64)
    else:
        eddy_params = np.ascontiguousarray(np.ravel(np.zeros(2)), np.float64)
    cdef np.ndarray[np.float64_t, ndim=1, mode="c"] eddy_params_c = eddy_params

    _run_kernel_diff_fixedN(&G_out, &N_out, &ddebug, verbose, N0, gmax, smax, TE, N_moments, &m_params_c[0], pns_thresh, T_readout, T_90, T_180, diffmode, dt_out, N_eddy, &eddy_params_c[0])

    G_return = np.zeros(N_out)
    for i in range(N_out):
        G_return[i] = G_out[i]

    debug_out = np.zeros(480000)
    for i in range(480000):
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

    N_moments = m_params.size//6
    m_params = np.ascontiguousarray(np.ravel(m_params), np.float64)
    cdef np.ndarray[np.float64_t, ndim=1, mode="c"] m_params_c = m_params

    cdef double *G_out
    cdef int N_out  
    cdef double *ddebug

    cdef np.ndarray[np.float64_t, ndim=1, mode="c"] G_in_c = np.ascontiguousarray(np.ravel(G_in), np.float64)

    N_eddy = len(eddy)//2
    if N_eddy > 0:
        eddy_params = np.ascontiguousarray(np.ravel(eddy), np.float64)
    else:
        eddy_params = np.ascontiguousarray(np.ravel(np.zeros(2)), np.float64)
    cdef np.ndarray[np.float64_t, ndim=1, mode="c"] eddy_params_c = eddy_params

    _run_kernel_diff_fixedN_Gin(&G_out, &N_out, &ddebug, verbose, &G_in_c[0], N0, gmax, smax, TE, N_moments, &m_params_c[0], pns_thresh, T_readout, T_90, T_180, diffmode, dt_out, N_eddy, &eddy_params_c[0])

    G_return = np.zeros(N_out)
    for i in range(N_out):
        G_return[i] = G_out[i]

    debug_out = np.zeros(480000)
    for i in range(480000):
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

    N_moments = m_params.size//6
    m_params = np.ascontiguousarray(np.ravel(m_params), np.float64)
    cdef np.ndarray[np.float64_t, ndim=1, mode="c"] m_params_c = m_params

    cdef double *G_out
    cdef int N_out  
    cdef double *ddebug

    N_eddy = len(eddy)//2
    if N_eddy > 0:
        eddy_params = np.ascontiguousarray(np.ravel(eddy), np.float64)
    else:
        eddy_params = np.ascontiguousarray(np.ravel(np.zeros(2)), np.float64)
    cdef np.ndarray[np.float64_t, ndim=1, mode="c"] eddy_params_c = eddy_params

    _run_kernel_diff_fixeddt(&G_out, &N_out, &ddebug, verbose, dt, gmax, smax, TE, N_moments, &m_params_c[0], pns_thresh, T_readout, T_90, T_180, diffmode, dt_out, N_eddy, &eddy_params_c[0])

    G_return = np.zeros(N_out)
    for i in range(N_out):
        G_return[i] = G_out[i]

    debug_out = np.zeros(480000)
    for i in range(480000):
        debug_out[i] = ddebug[i]

    return G_return, debug_out


def run_kernel_fixdt(gmax, smax, m_params, TE, T_readout, T_90, T_180, diffmode, dt = 0.4e-3, dt_out = -1.0, 
                         eddy = [], pns_thresh = -1.0, verbose = 1):

    m_params = np.array(m_params).flatten() 

    N_moments = m_params.size//6
    m_params = np.ascontiguousarray(np.ravel(m_params), np.float64)
    cdef np.ndarray[np.float64_t, ndim=1, mode="c"] m_params_c = m_params

    cdef double *G_out
    cdef int N_out  
    cdef double *ddebug

    N_eddy = len(eddy)//2
    if N_eddy > 0:
        eddy_params = np.ascontiguousarray(np.ravel(eddy), np.float64)
    else:
        eddy_params = np.ascontiguousarray(np.ravel(np.zeros(2)), np.float64)
    cdef np.ndarray[np.float64_t, ndim=1, mode="c"] eddy_params_c = eddy_params

    _run_kernel_diff_fixeddt(&G_out, &N_out, &ddebug, verbose, dt, gmax, smax, TE, N_moments, &m_params_c[0], pns_thresh, T_readout, T_90, T_180, diffmode, dt_out, N_eddy, &eddy_params_c[0])

    G_return = np.zeros(N_out) 
    for i in range(N_out):
        G_return[i] = G_out[i]

    debug_out = np.zeros(480000)
    for i in range(480000):
        debug_out[i] = ddebug[i]

    return G_return, debug_out