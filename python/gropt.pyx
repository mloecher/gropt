import numpy as np
cimport numpy as np
import time
cimport cython

cdef extern from "../src/wrappers.cpp":
    void _python_wrapper_v1 "python_wrapper_v1"(double *params0, double *params1, double **out0, double **out1, double **out2, int **outsize)


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

    print('Outsize0 =', outsize[0])
    # print(G_return)

    return G_return
