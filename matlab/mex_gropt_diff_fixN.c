#include <mex.h>
#include "matrix.h"
#include "optimize_kernel.c"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    double *temp;
    double *G;
    int N;
    double *ddebug;
    const mwSize *dims;
    
    int verbose = 0;
    
    double gmax = mxGetScalar(prhs[0]);
    double smax = mxGetScalar(prhs[1]);
    int MMT = mxGetScalar(prhs[2]);
    double TE = mxGetScalar(prhs[3]);
    double T_readout = mxGetScalar(prhs[4]);
    double T_90 = mxGetScalar(prhs[5]);
    double T_180 = mxGetScalar(prhs[6]);
    int N0 = mxGetScalar(prhs[7]);
    int diffmode = mxGetScalar(prhs[8]);
    
    double dt_out;
    if (nrhs > 9) {
        dt_out = mxGetScalar(prhs[9]);
    } else {
        dt_out = -1.0;
    }
    
    double pns_thresh;
    if (nrhs > 10) {
        pns_thresh = mxGetScalar(prhs[10]);
    } else {
        pns_thresh = -1.0;
    }
    
    int N_eddy;
    double *eddy_params;
    if (nrhs > 11) {
        eddy_params = mxGetPr(prhs[11]);
        dims = mxGetDimensions(prhs[11]);
        N_eddy = dims[1];
    } else {
        N_eddy = 0;
    }
    
    
    int ii = 0;
    double m_params[128]; // Make this big enough for anything
    memset(m_params, 0, 128*sizeof(double));
    int N_moments = 0;
    
    // M0 Nulling
    N_moments += 1;
    m_params[ii+0] = 0;
    m_params[ii+1] = 0;
    m_params[ii+2] = -1;
    m_params[ii+3] = -1;
    m_params[ii+4] = 0.0;
    m_params[ii+5] = 1.0e-3;
    ii += 6;
    
    // M1 Nulling
    if (MMT > 0) {
        N_moments += 1;
        m_params[ii+0] = 0;
        m_params[ii+1] = 1;
        m_params[ii+2] = -1;
        m_params[ii+3] = -1;
        m_params[ii+4] = 0.0;
        m_params[ii+5] = 1.0e-3;
        ii += 6;
    }
    
    // M2 Nulling
    if (MMT > 1) {
        N_moments += 1;
        m_params[ii+0] = 0;
        m_params[ii+1] = 2;
        m_params[ii+2] = -1;
        m_params[ii+3] = -1;
        m_params[ii+4] = 0.0;
        m_params[ii+5] = 1.0e-3;
        ii += 6;
    }
    
    run_kernel_diff_fixedN(&G, &N, &ddebug, verbose, N0, gmax, smax, TE, N_moments, m_params, pns_thresh, 
            T_readout, T_90, T_180, diffmode, dt_out, N_eddy, eddy_params);
    
    plhs[0] = mxCreateDoubleMatrix(1,N,mxREAL);

    // Why doesnt this work?: mxSetPr(plhs[0], G);
    temp = mxGetPr(plhs[0]);
    for (int i = 0; i < N; i++) {
        temp[i] = G[i];
    }
        

}