#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>

#include "optimize_kernel.h"



// This was the original TE finder, but it should be superceded by the openmp one
void minTE_diff(double **G_out, int *N_out, double **ddebug, int verbose,
                double dt0, double gmax, double smax, double search_bval,
                int N_moments, double *moments_params, double PNS_thresh, 
                double T_readout, double T_90, double T_180, int diffmode, double dt_out,
                int N_eddy, double *eddy_params, double slew_reg)
{
    // For testing purposes allow for verbose to be on only this function
    int verbose2;
    // verbose2 = verbose;
    verbose2 = 0;
    
    int N;
    double TE;
    double dt;
    double bval;
    double best_TE = 999.0;

    double T_lo = 0.0;
    double T_hi = 256.0;
    double T_range = T_hi-T_lo;

    // double dt = (TE-T_readout) * 1.0e-3 / (double) N;
    // double dt = dt0;

    N = 80;
    
    for (int i = 0; i < 6; i++) {
        TE = T_lo + (T_range)/2.0;
        dt = (TE-T_readout) * 1.0e-3 / (double) N;
        
        run_kernel_diff(G_out, N_out, ddebug, verbose2,
                        N, dt, gmax, smax, TE, 
                        N_moments, moments_params, PNS_thresh,
                        T_readout, T_90, T_180, diffmode,
                        1.0, 1.0, 100.0, 
                        10.0,  dt_out,
                        N_eddy, eddy_params,
                        0, NULL, search_bval,
                        0, NULL, slew_reg, 1);
        bval = (*ddebug)[13];
        
        if (verbose > 0) {printf ("Search at TE = %.2f gave bval = %.1f  T_range = %.2f\n", TE, bval, T_range);}
        if (bval > search_bval) {
            T_hi = TE;
            if (T_hi < best_TE) {
                best_TE = T_hi;
                if (verbose > 0) {printf ("Best TE = %.2f\n", best_TE);}
            }
        } else {
            T_lo = TE;
        }
        T_range = T_hi - T_lo;
        
    }
    
    if (verbose > 0) {printf ("\n***Second Round***\n\n");}


    T_lo = best_TE - 4.0;
    T_hi = best_TE + 4.0;
    T_range = T_hi-T_lo;

    dt = dt0;

    for (int i = 0; i < 6; i++) {
        TE = T_lo + (T_range)/2.0;
        N = round((TE-T_readout) * 1.0e-3/dt0);
        
        run_kernel_diff(G_out, N_out, ddebug, verbose2,
                        N, dt, gmax, smax, TE, 
                        N_moments, moments_params, PNS_thresh,
                        T_readout, T_90, T_180, diffmode,
                        1.0, 1.0, 100.0, 
                        10.0,  dt_out,
                        N_eddy, eddy_params,
                        0, NULL, search_bval,
                        0, NULL, slew_reg, 1);
        bval = (*ddebug)[13];
        
        if (verbose > 0) {printf ("Search at TE = %.2f gave bval = %.1f  T_range = %.2f\n", TE, bval, T_range);}
        if (bval > search_bval) {
            T_hi = TE;
            if (T_hi < best_TE) {
                best_TE = T_hi;
                if (verbose > 0) {printf ("Best TE = %.2f\n", best_TE);}
            }
        } else {
            T_lo = TE;
        }
        T_range = T_hi - T_lo;
        
    }

    if (verbose > 0) {printf ("\n***Final***\n\n");}

    TE = best_TE;
    N = round((TE-T_readout) * 1.0e-3/dt0);
    
    run_kernel_diff(G_out, N_out, ddebug, verbose2,
                    N, dt, gmax, smax, TE, 
                    N_moments, moments_params, PNS_thresh,
                    T_readout, T_90, T_180, diffmode,
                    1.0, 1.0, 100.0, 
                    10.0,  dt_out,
                    N_eddy, eddy_params,
                    0, NULL, -1.0,
                    0, NULL, slew_reg, 1);
    bval = (*ddebug)[13];
    
    if (verbose > 0) {printf ("Search at TE = %.2f gave bval = %.1f  T_range = %.2f\n", TE, bval, T_range);}

    // printf("\nN_out = %d\n", *N_out);
    // for (int i = 0; i < *N_out; i++) {
    //     printf("%.2f ", (*G_out)[i]);
    // }
    // printf("\n\n");
}


// This searches a TE range for the optimal timing, using openmp if available to do the search in parallel 
// i.e. each different Te being tested gets sent to a new thread job
void minTE_diff_par_worker(double *res, int N_delim, int N, double T_lo, double T_hi, int use_dt, 
                            double **G_out, int *N_out, int verbose,
                            double dt0, double gmax, double smax, double search_bval,
                            int N_moments, double *moments_params, double PNS_thresh, 
                            double T_readout, double T_90, double T_180, int diffmode, double dt_out,
                            int N_eddy, double *eddy_params, double slew_reg)
{
    // For testing purposes allow for verbose to be on only this function
    int verbose2;
    // verbose2 = verbose;
    verbose2 = 0;
    
    double T_range = T_hi-T_lo;
    
    double *bval_delim;
    bval_delim = (double*)malloc(N_delim * sizeof(double));

    double *TE_delim;
    TE_delim = (double*)malloc(N_delim * sizeof(double));

    int *N_out_delim;
    N_out_delim = (int*)malloc(N_delim * sizeof(int));

    double **par_all_G = (double **)malloc(N_delim * sizeof(double *));

    #pragma omp parallel for
    for (int i = 0; i < N_delim; i++) {
        double TE;
        double dt;
        double bval;
        int N_in;

        TE = T_lo + T_range * ((double) i / (double)(N_delim-1) );
        
        if (use_dt == 0) {
            dt = (TE-T_readout) * 1.0e-3 / (double) N;
            N_in = N;
        } else {
            dt = dt0;
            N_in = round((TE-T_readout) * 1.0e-3/dt0);
        }
        
        int par_N;
        double *par_debug;

        run_kernel_diff(&par_all_G[i], &par_N, &par_debug, verbose2,
                        N_in, dt, gmax, smax, TE, 
                        N_moments, moments_params, PNS_thresh,
                        T_readout, T_90, T_180, diffmode,
                        1.0, 1.0, 100.0, 
                        10.0,  dt_out,
                        N_eddy, eddy_params,
                        0, NULL, search_bval,
                        0, NULL, slew_reg, 1);
        
        bval = par_debug[13];

        N_out_delim[i] = par_N;
        bval_delim[i] = bval;
        TE_delim[i] = TE;

        free(par_debug);
        
        // if (verbose > 0) {printf ("Search at TE = %.2f gave bval = %.1f  T_range = %.2f\n", TE, bval, T_range);}        
    }

    for (int i = 0; i < N_delim; i++) {
        if (verbose > 0) {printf("   %.2f  %.2f\n", TE_delim[i], bval_delim[i]);}
    }

    double best_TE = 99999.0;
    double min_diff = 99999.0;
    double best_diff = 99999.0;
    int best_ind = -1;
    double diff;
    for (int i = 0; i < N_delim; i++) {
        diff = bval_delim[i] - search_bval;
        if (fabs(diff) < min_diff) {
            best_TE = TE_delim[i];
            min_diff = fabs(diff);
            best_diff = diff;
            best_ind = i;
        }
    }
    if (verbose > 0) {printf(" *** Best TE = %.2f    (diff = %.2f)\n", best_TE, best_diff);}
    res[0] = best_TE;
    res[1] = best_diff;

    

    *N_out = N_out_delim[best_ind];
    *G_out = par_all_G[best_ind];

    for (int i = 0; i < N_delim; i++) {
        if (i != best_ind) {free(par_all_G[i]);}
    }

    free(N_out_delim);
    free(TE_delim);
    free(bval_delim);
}


// Find the optimal TE using the parallelized TE worker
void minTE_diff_par(double **G_out, int *N_out, double **ddebug, int verbose,
                double dt0, double gmax, double smax, double search_bval,
                int N_moments, double *moments_params, double PNS_thresh, 
                double T_readout, double T_90, double T_180, int diffmode, double dt_out,
                int N_eddy, double *eddy_params, double slew_reg)
{
    int N_delim1, N_delim2, N;
    double T_lo, T_hi;
    double dTE;
    double res[3];
    double TE_expand = 2.0;

    // How many TEs to check with each loop
    N_delim1 = 8;
    N_delim2 = 8;
    N = 64;
    T_lo = 2.0;
    T_hi = 256.0;

    double *G_out_temp;

    int N_loop1 = 20; // This is old, now it is set high and the loop breaks when dTE is small enough
    for (int i = 0; i < N_loop1; i++) {
        minTE_diff_par_worker(res, N_delim1, N, T_lo, T_hi, 0, &G_out_temp, N_out, verbose, dt0, gmax, smax, search_bval, 
                            N_moments, moments_params, PNS_thresh, 
                                T_readout, T_90, T_180, diffmode, dt_out, N_eddy, eddy_params, 1.0);
        
        dTE = (T_hi-T_lo) / (double)(N_delim1-1);
        if (res[1] > 0) {
            T_lo = res[0]-dTE;
            T_hi = res[0];
        } else {
            T_lo = res[0];
            T_hi = res[0]+dTE;
        }
        
        free(G_out_temp); // We only need the last one of these, but storing and freeing them shouldnt take too long
        G_out_temp = NULL; // code laziness, we free a second time later

        // What would the next loop dTE be if we stopped now?
        double r2dTE = (T_hi - T_lo + TE_expand) / (double)(N_delim2-1);

        if (r2dTE <= (1.0e3*dt0)) {
            break;
        } else if (r2dTE < 1.1*TE_expand) {
            break;
        }
        
    }

    // Most likely we are increasing N, so out bvals will go up slightly, so reduce T_lo to account for this:
    if (T_lo > 5.0) {T_lo -= TE_expand;}
    
    // This second set of loops used the native dt
    
    int N_loop2 = 20; // This is old, now it is set high and the loop breaks when dTE is less than dt0
    for (int i = 0; i < N_loop2; i++) {
        free(G_out_temp);
        
        minTE_diff_par_worker(res, N_delim2, N, T_lo, T_hi, 1, &G_out_temp, N_out, verbose, dt0, gmax, smax, search_bval, 
                              N_moments, moments_params, PNS_thresh, 
                              T_readout, T_90, T_180, diffmode, dt_out, N_eddy, eddy_params, 1.0);       
        
        dTE = (T_hi-T_lo) / (double)(N_delim2-1);
        
        if (res[1] > 0) {
            T_lo = res[0]-dTE;
            T_hi = res[0];
        } else {
            T_lo = res[0];
            T_hi = res[0]+dTE;
        }
        
        *G_out = G_out_temp;

        if (dTE <= (2.0e3*dt0)) {
            break;
        }
    }

    // printf("\nN_out = %d\n", *N_out);
    // for (int i = 0; i < *N_out; i++) {
    //     printf("%.2f ", (*G_out)[i]);
    // }
    // printf("\n\n");

}
