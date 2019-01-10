#include <mex.h>
#include "matrix.h"
#include "optimize_kernel.c"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    double *temp;
    double *G;
    int N;
    double *ddebug;
    
    double gmax = mxGetScalar(prhs[0]);
    double smax = mxGetScalar(prhs[1]);
    int MMT = mxGetScalar(prhs[2]);
    double TE = mxGetScalar(prhs[3]);
    double T_readout = mxGetScalar(prhs[4]);
    double T_90 = mxGetScalar(prhs[5]);
    double T_180 = mxGetScalar(prhs[6]);
    int N0 = mxGetScalar(prhs[7]);
    int diffmode = mxGetScalar(prhs[8]);
    
    double m_tol[3]={0.0, -1.0, -1.0};
    if (MMT > 0) {m_tol[1] = 0.0;}
    if (MMT > 1) {m_tol[2] = 0.0;}
    
    run_kernel_diff_fixedN(&G, &N, &ddebug, N0, gmax, smax, m_tol, TE, T_readout, T_90, T_180, diffmode, -1.0);
    
    plhs[0] = mxCreateDoubleMatrix(1,N,mxREAL);

    // Why doesnt this work?: mxSetPr(plhs[0], G);
    temp = mxGetPr(plhs[0]);
    for (int i = 0; i < N; i++) {
        temp[i] = G[i];
    }
        

}