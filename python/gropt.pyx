import numpy as np
cimport numpy as np
import time
cimport cython

cdef extern from "../src/wrappers.cpp":
    void _python_wrapper_v1 "python_wrapper_v1"(double *params0, double *params1, double **out0, double **out1, double **out2, int **outsize)

    void _threed_diff "threed_diff"(double *params0, double *params1, double **out0, double **out1, double **out2, int **outsize)

    void _gropt_legacy "gropt_legacy"(double **G_out, int *N_out, double **ddebug, int verbose, 
                                        double dt0, double gmax, double smax, double TE, 
                                        int N_moments, double *moments_params, double PNS_thresh, 
                                        double T_readout, double T_90, double T_180, int diffmode, double dt_out,
                                        int N_eddy, double *eddy_params, double search_bval, double slew_reg, int Naxis)

    void _diff_duty_cycle "diff_duty_cycle"(double dt, double T_90, double T_180, double T_readout, double TE, 
                                        int N_moments, double gmax, double smax, double bval, double duty_cycle,
                                        double **out0, double **out1, double **out2, int **outsize) 

    # void _spect_phase_contrast "spect_phase_contrast"(double dt, double T, double gmax, double smax, int Neddy,
    #                                                   double **out0, double **out1, double **out2, int **outsize)

    void _rewinder3 "rewinder3"(double dt, double T, double gmax, double smax, double *G0_in, double *M0_in,
                                double l2_weight,
                                double **out0, double **out1, double **out2, int **outsize)

    void _acoustic_v1 "acoustic_v1"(double dt, double T, double gmax, double smax, double *G0_in, double *H_in,
                double l2_weight, double a_weight, int verbose,
                double **out0, double **out1, double **out2, int **outsize) 

    void _acoustic_v2 "acoustic_v2"(double dt, double T, double gmax, double smax, double *G0_in, double complex *H_in,
                double l2_weight, double a_weight, int verbose,
                double **out0, double **out1, double **out2, int **outsize) 

    void _acoustic_v3 "acoustic_v3"(double dt, double T, double gmax, double smax, double *G0_in, 
                double complex *H_in, int N_H,
                int N_moments, double *moments_params,
                double l2_weight, double a_weight, int verbose,
                int p_iter, int p_cg_iter, int p_obj_min, double p_obj_scale,
                double **out0, double **out1, double **out2, int **outsize)

    void _girf_ec_v1 "girf_ec_v1"(double dt, double T, double gmax, double smax, double *G0_in, 
                double complex *H_in, double *girf_win_in, int N_H,
                int N_moments, double *moments_params,
                double l2_weight, double a_weight, int verbose,
                int p_iter, int p_cg_iter, int p_obj_min, double p_obj_scale,
                double **out0, double **out1, double **out2, int **outsize)

    void _girf_ec_v2 "girf_ec_v2"(double dt, double T, double gmax, double smax, double *G0_in, 
                double complex *H_in, double *girf_win_in, int N_H,
                int N_moments, double *moments_params,
                double l2_weight, double a_weight, double stim_thresh, int verbose,
                int p_iter, int p_cg_iter, int p_obj_min, double p_obj_scale,
                double **out0, double **out1, double **out2, int **outsize)

    void _accel_test_v1 "accel_test_v1"(double dt, double T, double gmax, double smax, double M1, double M2,
                   double l2_weight, int do_m2, int verbose,
                   double **out0, double **out1, double **out2, int **outsize)

    void _girf_tester_v1 "girf_tester_v1"(int N, double *G, double complex *H_in, 
                double complex **out0, double complex **out1, double complex **out2, int **outsize)

    void _simple_bipolar_pns "simple_bipolar_pns"(double dt, double T, double gmax, double smax, double M0, double M1,
                                                double stim_thresh, double l2_weight, int verbose,
                                                double **out0, double **out1, double **out2, int **outsize); 

    void _cones_pns_3 "cones_pns_3"(double dt, int N, double gmax, double smax, double *G0_in,
                                    int N_moments, double *moments_in,
                                    double eddy_lam, int eddy_stop, double eddy_tol,
                                    double stim_thresh,  double l2_weight, int rot_var_mode, int verbose,
                                    int p_iter, int p_cg_iter, int p_obj_min, double p_obj_scale,
                                    double **out0, double **out1, double **out2, int **outsize)

    void _cones_pns_3_v2 "cones_pns_3_v2"(double dt, int N, double gmax, double smax, double *G0_in,
                                    int N_moments, double *moments_in,
                                    double *eddy_lam, int eddy_stop, double eddy_tol, int Nlam,
                                    double stim_thresh,  double l2_weight, int rot_var_mode, int verbose,
                                    int p_iter, int p_cg_iter, int p_obj_min, double p_obj_scale,
                                    double **out0, double **out1, double **out2, int **outsize)


    void _diff_pre_eddy "diff_pre_eddy"(double dt, double T_90, double T_180, double T_readout, double T_pre, double TE, 
                                    int moment_order, double gmax, double smax, double *eddy_lam_in, int Nlam, double maxwell_tol, double b_weight,
                                    double **out0, double **out1, double **out2, int **outsize)

def array_prep(A, dtype, linear=True):
    if not A.flags['C_CONTIGUOUS']:
        A = np.ascontiguousarray(A)
    
    A = A.astype(dtype, order='C', copy=False)
    
    if linear:
        A = A.ravel()

    return A 



@cython.boundscheck(False) 
@cython.wraparound(False)
def diff_pre_eddy(dt, T_90, T_180, T_readout, T_pre, TE, moment_order, gmax, smax, eddy_lam, maxwell_tol, b_weight = -1):

    cdef double *out0
    cdef double *out1
    cdef double *out2
    cdef int *outsize

    if np.isscalar(eddy_lam):
        eddy_lam = [eddy_lam,]
    eddy_lam = np.array(eddy_lam)
    cdef double[::1] eddy_lam_view = array_prep(eddy_lam, np.float64)

    Nlam = eddy_lam.size

    _diff_pre_eddy(dt, T_90, T_180, T_readout, T_pre, TE, 
                   moment_order, gmax, smax, &eddy_lam_view[0], Nlam, maxwell_tol, b_weight,
                   &out0, &out1, &out2, &outsize)

    print('Done with func and in .pyx', flush=True)
    print('outsize[0]', outsize[0], flush=True)
    G_return = np.empty(outsize[0])
    for i in range(outsize[0]):
        G_return[i] = out0[i]
    print('Done with func and in .pyx set G_return', flush=True)

    return G_return

@cython.boundscheck(False) 
@cython.wraparound(False)
def cones_pns_3(N, gmax, smax, G0, moments, stim_thresh = -1, 
                eddy_lam = 80, eddy_stop = -1, eddy_tol = -1,
                rot_var_mode = 1, dt = 10e-6, l2_weight = -1, verbose = 0,
                p_iter = 500, p_cg_iter = 10, p_obj_min = 0, p_obj_scale = 1.0):

    cdef double *out0
    cdef double *out1
    cdef double *out2
    cdef int *outsize

    cdef double[::1] G0_view = array_prep(G0, np.float64)
    N_moments = moments.shape[0]
    cdef double[::1] moments_view = array_prep(moments, np.float64)

    _cones_pns_3(dt, N, gmax, smax, &G0_view[0], 
                N_moments, &moments_view[0], 
                eddy_lam, eddy_stop, eddy_tol,
                stim_thresh, l2_weight, rot_var_mode, verbose,
                p_iter, p_cg_iter, p_obj_min, p_obj_scale,
                &out0, &out1, &out2, &outsize)

    np_out0 = np.empty(outsize[0], float)
    for i in range(outsize[0]):
        np_out0[i] = out0[i]

    np_out1 = np.empty(outsize[1], float)
    for i in range(outsize[1]):
        np_out1[i] = out1[i]

    return np_out0, np_out1



@cython.boundscheck(False) 
@cython.wraparound(False)
def cones_pns_3_v2(N, gmax, smax, G0, moments, stim_thresh = -1, 
                eddy_lam = [80,], eddy_stop = -1, eddy_tol = -1,
                rot_var_mode = 1, dt = 10e-6, l2_weight = -1, verbose = 0,
                p_iter = 500, p_cg_iter = 10, p_obj_min = 0, p_obj_scale = 1.0):

    cdef double *out0
    cdef double *out1
    cdef double *out2
    cdef int *outsize

    cdef double[::1] G0_view = array_prep(G0, np.float64)
    N_moments = moments.shape[0]
    cdef double[::1] moments_view = array_prep(moments, np.float64)
    
    eddy_lam = np.array(eddy_lam)
    N_lam = eddy_lam.size
    cdef double[::1] eddy_lam_view = array_prep(eddy_lam, np.float64)

    _cones_pns_3_v2(dt, N, gmax, smax, &G0_view[0], 
                N_moments, &moments_view[0], 
                &eddy_lam_view[0], eddy_stop, eddy_tol, N_lam,
                stim_thresh, l2_weight, rot_var_mode, verbose,
                p_iter, p_cg_iter, p_obj_min, p_obj_scale,
                &out0, &out1, &out2, &outsize)

    np_out0 = np.empty(outsize[0], float)
    for i in range(outsize[0]):
        np_out0[i] = out0[i]

    np_out1 = np.empty(outsize[1], float)
    for i in range(outsize[1]):
        np_out1[i] = out1[i]

    return np_out0, np_out1

@cython.boundscheck(False) 
@cython.wraparound(False)
def simple_bipolar_pns(dt, T, gmax, smax, M0, M1, stim_thresh, l2_weight = -1, verbose = 0):

    cdef double *out0
    cdef double *out1
    cdef double *out2
    cdef int *outsize


    _simple_bipolar_pns(dt, T, gmax, smax, M0, M1, stim_thresh, l2_weight, verbose, &out0, &out1, &out2, &outsize)

    np_out0 = np.empty(outsize[0], float)
    for i in range(outsize[0]):
        np_out0[i] = out0[i]

    np_out1 = np.empty(outsize[1], float)
    for i in range(outsize[1]):
        np_out1[i] = out1[i]


    return np_out0, np_out1


@cython.boundscheck(False) 
@cython.wraparound(False)
def girf_tester_v1(N, G, H_in):

    cdef double complex *out0
    cdef double complex *out1
    cdef double complex *out2
    cdef int *outsize

    cdef double[::1] G_view = array_prep(G, np.float64)
    cdef double complex[::1] H_in_view = array_prep(H_in, np.complex128)

    _girf_tester_v1(N, &G_view[0], &H_in_view[0], &out0, &out1, &out2, &outsize)

    np_out0 = np.empty(outsize[0], complex)
    for i in range(outsize[0]):
        np_out0[i] = out0[i]

    return np_out0


@cython.boundscheck(False) 
@cython.wraparound(False)
def accel_test_v1(dt, T, gmax, smax, M1, M2, l2_weight = 1e-6, verbose=0):

    cdef double *out0
    cdef double *out1
    cdef double *out2
    cdef int *outsize

    if M2 is None:
        M2 = 0.0
        do_m2 = 0
    else:
        do_m2 = 1

    _accel_test_v1(dt, T, gmax, smax, M1, M2, l2_weight, do_m2, verbose, &out0, &out1, &out2, &outsize)

    G_return = np.empty(outsize[0])
    for i in range(outsize[0]):
        G_return[i] = out0[i]

    debug_return = np.empty(outsize[1])
    for i in range(outsize[1]):
        debug_return[i] = out1[i]

    return G_return, debug_return


@cython.boundscheck(False) 
@cython.wraparound(False)
def acoustic_v1(dt, T, gmax, smax, G0, H0, l2_weight, a_weight = 0.0, verbose=0):
    print('Calling acoustic_v1')

    cdef double[::1] G0_view = array_prep(G0, np.float64)
    cdef double[::1] H0_view = array_prep(H0, np.float64)

    cdef double *out0
    cdef double *out1
    cdef double *out2
    cdef int *outsize

    _acoustic_v1(dt, T, gmax, smax, &G0_view[0], &H0_view[0], l2_weight, a_weight, verbose, &out0, &out1, &out2, &outsize)

    G_return = np.empty(outsize[0])
    for i in range(outsize[0]):
        G_return[i] = out0[i]

    debug_return = np.empty(outsize[1])
    for i in range(outsize[1]):
        debug_return[i] = out1[i]

    interp_return = np.empty(outsize[2])
    for i in range(outsize[2]):
        interp_return[i] = out2[i]

    return G_return, debug_return, interp_return


@cython.boundscheck(False) 
@cython.wraparound(False)
def acoustic_v2(dt, T, gmax, smax, G0, H0, l2_weight, a_weight = 0.0, verbose=0):
    print('Calling acoustic_v2')

    cdef double[::1] G0_view = array_prep(G0, np.float64)
    cdef double complex[::1] H0_view = array_prep(H0, np.complex128)

    cdef double *out0
    cdef double *out1
    cdef double *out2
    cdef int *outsize

    _acoustic_v2(dt, T, gmax, smax, &G0_view[0], &H0_view[0], l2_weight, a_weight, verbose, &out0, &out1, &out2, &outsize)

    print('outsize', outsize[0], outsize[1], outsize[2])

    G_return = np.empty(outsize[0])
    for i in range(outsize[0]):
        G_return[i] = out0[i]

    debug_return = np.empty(outsize[1])
    for i in range(outsize[1]):
        debug_return[i] = out1[i]

    interp_return = np.empty(outsize[2])
    for i in range(outsize[2]):
        interp_return[i] = out2[i]

    return G_return, debug_return, interp_return


@cython.boundscheck(False) 
@cython.wraparound(False)
def acoustic_v3(dt, T, gmax, smax, G0, H0, moment_params, l2_weight, a_weight = 0.0, verbose=0,
                p_iter = 500, p_cg_iter = 10, p_obj_min = 0, p_obj_scale = 1.0):
    # print('Calling acoustic_v3')

    cdef double[::1] G0_view = array_prep(G0, np.float64)

    cdef int N_H = H0.size
    cdef double complex[::1] H0_view = array_prep(H0, np.complex128)

    cdef int N_moments = moment_params.shape[0]
    cdef double[::1] moment_params_view = array_prep(moment_params, np.float64)

    cdef double *out0
    cdef double *out1
    cdef double *out2
    cdef int *outsize

    _acoustic_v3(dt, T, gmax, smax, &G0_view[0], 
                &H0_view[0], N_H,
                N_moments, &moment_params_view[0],
                l2_weight, a_weight, verbose, 
                p_iter, p_cg_iter, p_obj_min, p_obj_scale,
                &out0, &out1, &out2, &outsize)

    # print('outsize', outsize[0], outsize[1], outsize[2])

    G_return = np.empty(outsize[0])
    for i in range(outsize[0]):
        G_return[i] = out0[i]

    debug_return = np.empty(outsize[1])
    for i in range(outsize[1]):
        debug_return[i] = out1[i]

    interp_return = np.empty(outsize[2])
    for i in range(outsize[2]):
        interp_return[i] = out2[i]

    return G_return, debug_return, interp_return


@cython.boundscheck(False) 
@cython.wraparound(False)
def girf_ec_v1(dt, T, gmax, smax, G0, H0, girf_win, moment_params, l2_weight, a_weight = 0.0, verbose=0,
                p_iter = 500, p_cg_iter = 10, p_obj_min = 0, p_obj_scale = 1.0):
    # print('Calling acoustic_v3')

    cdef double[::1] G0_view = array_prep(G0, np.float64)

    cdef int N_H = H0.size
    cdef double complex[::1] H0_view = array_prep(H0, np.complex128)

    cdef double[::1] girf_win_view = array_prep(girf_win, np.float64)

    cdef int N_moments = moment_params.shape[0]
    cdef double[::1] moment_params_view = array_prep(moment_params, np.float64)

    cdef double *out0
    cdef double *out1
    cdef double *out2
    cdef int *outsize

    _girf_ec_v1(dt, T, gmax, smax, &G0_view[0], 
                &H0_view[0], &girf_win_view[0], N_H,
                N_moments, &moment_params_view[0],
                l2_weight, a_weight, verbose, 
                p_iter, p_cg_iter, p_obj_min, p_obj_scale,
                &out0, &out1, &out2, &outsize)

    # print('outsize', outsize[0], outsize[1], outsize[2])

    G_return = np.empty(outsize[0])
    for i in range(outsize[0]):
        G_return[i] = out0[i]

    debug_return = np.empty(outsize[1])
    for i in range(outsize[1]):
        debug_return[i] = out1[i]

    interp_return = np.empty(outsize[2])
    for i in range(outsize[2]):
        interp_return[i] = out2[i]

    return G_return, debug_return, interp_return


@cython.boundscheck(False) 
@cython.wraparound(False)
def girf_ec_v2(dt, T, gmax, smax, G0, H0, girf_win, moment_params, l2_weight, a_weight = 0.0, stim_thresh = -1, verbose=0,
                p_iter = 500, p_cg_iter = 10, p_obj_min = 0, p_obj_scale = 1.0):
    # print('Calling acoustic_v3')

    cdef double[::1] G0_view = array_prep(G0, np.float64)

    cdef int N_H = H0.size
    cdef double complex[::1] H0_view = array_prep(H0, np.complex128)

    cdef double[::1] girf_win_view = array_prep(girf_win, np.float64)

    cdef int N_moments = moment_params.shape[0]
    cdef double[::1] moment_params_view = array_prep(moment_params, np.float64)

    cdef double *out0
    cdef double *out1
    cdef double *out2
    cdef int *outsize

    _girf_ec_v2(dt, T, gmax, smax, &G0_view[0], 
                &H0_view[0], &girf_win_view[0], N_H,
                N_moments, &moment_params_view[0],
                l2_weight, a_weight, stim_thresh, verbose, 
                p_iter, p_cg_iter, p_obj_min, p_obj_scale,
                &out0, &out1, &out2, &outsize)

    # print('outsize', outsize[0], outsize[1], outsize[2])

    G_return = np.empty(outsize[0])
    for i in range(outsize[0]):
        G_return[i] = out0[i]

    debug_return = np.empty(outsize[1])
    for i in range(outsize[1]):
        debug_return[i] = out1[i]

    interp_return = np.empty(outsize[2])
    for i in range(outsize[2]):
        interp_return[i] = out2[i]

    return G_return, debug_return, interp_return

@cython.boundscheck(False) 
@cython.wraparound(False)
def rewinder3(dt, T, gmax, smax, G0, M0, l2_weight, verbose=0):

    cdef double[::1] G0_view = array_prep(G0, np.float64)

    cdef double[::1] M0_view = array_prep(M0, np.float64)

    cdef double *out0
    cdef double *out1
    cdef double *out2
    cdef int *outsize

    _rewinder3(dt, T, gmax, smax, &G0_view[0], &M0_view[0], l2_weight, &out0, &out1, &out2, &outsize)

    G_return = np.empty(outsize[0])
    for i in range(outsize[0]):
        G_return[i] = out0[i]

    debug_return = np.empty(outsize[1])
    for i in range(outsize[1]):
        debug_return[i] = out1[i]

    # interp_return = np.empty(outsize[2])
    # for i in range(outsize[2]):
    #     interp_return[i] = out2[i]

    return G_return, debug_return



@cython.boundscheck(False) 
@cython.wraparound(False)
def threed_diff(params0, verbose=0):

    # params0 = np.zeros(100)
    cdef double[::1] params0_view = array_prep(params0, np.float64)

    params1 = np.zeros(100)
    cdef double[::1] params1_view = array_prep(params1, np.float64)

    cdef double *out0
    cdef double *out1
    cdef double *out2
    cdef int *outsize

    _threed_diff(&params0_view[0], &params1_view[0], &out0, &out1, &out2, &outsize)

    G_return = np.empty(outsize[0])
    for i in range(outsize[0]):
        G_return[i] = out0[i]

    # debug_return = np.empty(outsize[1])
    # for i in range(outsize[1]):
    #     debug_return[i] = out1[i]

    # interp_return = np.empty(outsize[2])
    # for i in range(outsize[2]):
    #     interp_return[i] = out2[i]

    return G_return

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

    interp_return = np.empty(outsize[2])
    for i in range(outsize[2]):
        interp_return[i] = out2[i]

    return G_return, debug_return, interp_return

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



@cython.boundscheck(False) 
@cython.wraparound(False)
def diff_duty_cycle(dt, T_90, T_180, T_readout, TE, 
                N_moments, gmax, smax, bval, duty_cycle, verbose=0):



    cdef double *out0
    cdef double *out1
    cdef double *out2
    cdef int *outsize

    
    _diff_duty_cycle(dt, T_90, T_180, T_readout, TE, 
                     N_moments, gmax, smax, bval, duty_cycle, &out0, &out1, &out2, &outsize)

    G_return = np.empty(outsize[0])
    for i in range(outsize[0]):
        G_return[i] = out0[i]

    debug_return = np.empty(outsize[1])
    for i in range(outsize[1]):
        debug_return[i] = out1[i]

    return G_return, debug_return



# @cython.boundscheck(False) 
# @cython.wraparound(False)
# def spect_phase_contrast(dt, T, gmax, smax, Neddy, verbose=0):

#     cdef double *out0
#     cdef double *out1
#     cdef double *out2
#     cdef int *outsize

#     _spect_phase_contrast(dt, T, gmax, smax, Neddy, &out0, &out1, &out2, &outsize)

#     G_return = np.empty(outsize[0])
#     for i in range(outsize[0]):
#         G_return[i] = out0[i]

#     debug_return = np.empty(outsize[1])
#     for i in range(outsize[1]):
#         debug_return[i] = out1[i]

#     spect_return = np.empty(outsize[2])
#     for i in range(outsize[2]):
#         spect_return[i] = out2[i]


#     return G_return, debug_return, spect_return