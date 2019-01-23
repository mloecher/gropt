#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include "cvx_matrix.h"
#include "op_slewrate.h"
#include "op_moments.h"
#include "op_eddy.h"
#include "op_beta.h"
#include "op_bval.h"
#include "op_gradient.h"
#include "op_pns.h"

#define EDDY_PARAMS_LEN 2
#define MOMENTS_PARAMS_LEN 6

void cvx_optimize_kernel(cvx_mat *G, cvxop_gradient *opG, cvxop_slewrate *opD, 
                        cvxop_moments *opQ, cvxop_eddy *opE, cvxop_beta *opC, 
                        cvxop_bval *opB, cvxop_pns *opP,
                         int N, double relax, int verbose, double bval_reduction, double *ddebug, int N_converge, double stop_increase)
{    
    int max_iter = 30000;
    int check_amount = 100;
    int i_check = 0;
    int N_backlog = max_iter / check_amount;
    double *bval_backlog = (double *)malloc(N_backlog*sizeof(double));
    for (int i = 0; i < N_backlog; i++) {
        bval_backlog[i] = 0.0;
    }

    int converge_count = 0;
    int limit_count = 0;

    int rebalance_count = 0;

    for (int i = 0; i < opB->zB.N; i++) {
        opB->zB.vals[i] = 0.0;
    } 
    for (int i = 0; i < opD->zD.N; i++) {
        opD->zD.vals[i] = 0.0;
    }
    for (int i = 0; i < opQ->zQ.N; i++) {
        opQ->zQ.vals[i] = 0.0;
    }

    cvx_mat xbar;
    copyNewMatrix(G, &xbar);
    cvxmat_setvals(&xbar, 0.0);

    cvx_mat taumx;
    copyNewMatrix(G, &taumx);
    cvxmat_setvals(&taumx, 0.0);

    cvx_mat txmx;
    copyNewMatrix(G, &txmx);
    cvxmat_setvals(&txmx, 0.0);

    // tau scaling vector
    cvx_mat tau;
    copyNewMatrix(G, &tau);
    cvxmat_setvals(&tau, 0.0);

    cvxop_slewrate_add2tau(opD, &tau);
    cvxop_moments_add2tau(opQ, &tau);
    cvxop_eddy_add2tau(opE, &tau);
    cvxop_bval_add2tau(opB, &tau);
    cvxop_beta_add2tau(opC, &tau);
    cvxop_pns_add2tau(opP, &tau);
    cvxmat_EWinvert(&tau);

    double obj0 = 1.0;
    double obj1 = 1.0;    

    int count = 0;
    while (count < max_iter) {
    
        // xbar = G-tau.*((D'*zD)+(Q'*zQ)+C'+(B'*zB))
        cvxmat_setvals(&taumx, 0.0);

        cvxop_slewrate_add2taumx(opD, &taumx);
        cvxop_moments_add2taumx(opQ, &taumx);
        cvxop_eddy_add2taumx(opE, &taumx);
        cvxop_beta_add2taumx(opC, &taumx);
        cvxop_bval_add2taumx(opB, &taumx);
        cvxop_pns_add2taumx(opP, &taumx);

        cvxmat_EWmultIP(&taumx, &tau);

        cvxmat_subractMat(&xbar, G, &taumx);

        // xbar = gradient_limits(xbar)
        cvxop_gradient_limiter(opG, &xbar);

        // if (opB->active > 0 ) {
        //     cvxop_bval_proxxbar(opB, &xbar, eps);
        // }

        // txmx = 2*xbar-G;
        cvxmat_subractMatMult1(&txmx, 2.0, &xbar, G);


        // zDbuff  = zD + sigD.*(D*txmx);
        // zQbuff  = zQ + sigQ.*(Q*txmx);

        // zDbar = zDbuff - sigD.*min(SRMAX,max(-SRMAX,zDbuff./sigD));
        // zQbar = zQbuff - sigQ.*min(mvec,max(-mvec,zQbuff./sigQ));

        // zD=p*zDbar+(1-p)*zD;
        // zQ=p*zQbar+(1-p)*zQ;
        cvxop_slewrate_update(opD, &txmx, relax);
        cvxop_moments_update(opQ, &txmx, relax);
        cvxop_eddy_update(opE, &txmx, relax);
        cvxop_bval_update(opB, &txmx, relax, ddebug, count);
        cvxop_pns_update(opP, &txmx, relax);

        // G=p*xbar+(1-p)*G;
        cvxmat_updateG(G, relax, &xbar);

        // Need checks here
        if ( count % 100 == 0 ) {


            obj1 = cvxop_gradient_getbval(opG, G);
            
            bval_backlog[i_check] = obj1;
            
            double ii_backlog = 0;
            double bval_backlog0 = 0.0;
            for (int i = 0; i < 5; i++) {
                if ((i_check - i) >= 0) {
                    bval_backlog0 += bval_backlog[ (i_check - i) ];
                } else {
                    bval_backlog0 += 999999999999.0;
                }
                ii_backlog += 1.0;
            }
            bval_backlog0 /= ii_backlog;
            

            ii_backlog = 0;
            double bval_backlog1 = 0.0;
            for (int i = 5; i < 10; i++) {
                if ((i_check - i) >= 0) {
                    bval_backlog1 += bval_backlog[ (i_check - i) ];
                } else {
                    bval_backlog1 += 9999999999.0;
                }
                ii_backlog += 1.0;
            }
            bval_backlog1 /= ii_backlog;

            i_check += 1;

            double backlog_diff = (sqrt(bval_backlog0) - sqrt(bval_backlog1)) / sqrt(bval_backlog1);

            bval_backlog0 = (sqrt(obj1) - sqrt(bval_backlog1)) / sqrt(obj1);
            bval_backlog1 = (sqrt(obj1) - sqrt(bval_backlog1)) / sqrt(obj1);

            int is_converged = 0;
            if (fabs(backlog_diff) < stop_increase) {
                is_converged = 1;
            }

            rebalance_count += 1;


            if (verbose>0) {printf("\ncount = %d   rc = %d   obj = %.1f  backlog_diff = %.2e\n", count, rebalance_count, obj1, backlog_diff);}
            int bad_slew = cvxop_slewrate_check(opD, G);
            int bad_moments = cvxop_moments_check(opQ, G);
            int bad_gradient = cvxop_gradient_check(opG, G);
            int bad_eddy = cvxop_eddy_check(opE, G);
            cvxop_pns_check(opP, G);

            int limit_break = 0;
            limit_break += bad_slew;
            limit_break += bad_moments;
            limit_break += bad_gradient;
            limit_break += bad_eddy;

            if ( (count > 0) && (rebalance_count > N_converge)  && (is_converged > 0) && (limit_break == 0)) {
                if (verbose > 0) {
                    printf("** Early termination at count = %d   bval = %.1f\n", count, obj1);
                }
                break;
            }


           int needs_rebalancing = 0;
           if ( (is_converged > 0) && (rebalance_count > N_converge) ) {
               needs_rebalancing = 1;
           }

            if ( (bval_reduction > 0.0) && (count > 0) && (needs_rebalancing > 0) ) {

            
                rebalance_count = 0;

                if (verbose > 0) {
                    printf("\n\n !-!-!-!-!-!-! Converged to an inadequate waveform, reweighting !-!-!-!-!-!-! \n");
                } 

                if (bad_moments > 0) {
                    cvxop_moments_reweight(opQ, bval_reduction);
                    if (verbose > 0) {printf("  ^^ moments ^^  ");}
                }
                if (bad_slew > 0) {
                    cvxop_slewrate_reweight(opD, bval_reduction);
                    if (verbose > 0) {printf("  ^^ slew ^^  ");}
                }
                if (bad_eddy > 0) {
                    cvxop_eddy_reweight(opE, bval_reduction);
                    if (verbose > 0) {printf("  ^^ eddy ^^  ");}
                }
                
                if ((bad_slew < 1) && (bad_moments < 1) && (bad_eddy < 1)) {
                    cvxop_bval_reweight(opB, bval_reduction);
                    cvxop_beta_reweight(opC, bval_reduction);
                    if (verbose > 0) {printf("  ^^ bval ^^  ");}
                }                

                if (verbose > 0) { printf("\n");}

                cvxmat_setvals(&tau, 0.0);
                cvxop_slewrate_add2tau(opD, &tau);
                cvxop_moments_add2tau(opQ, &tau);
                cvxop_eddy_add2tau(opE, &tau);
                cvxop_bval_add2tau(opB, &tau);
                cvxop_beta_add2tau(opC, &tau);
                cvxop_pns_add2tau(opP, &tau);
                cvxmat_EWinvert(&tau);
                
                for (int i = 0; i < opB->zB.N; i++) {
                    opB->zB.vals[i] = 0.0; 
                }
                for (int i = 0; i < opD->zD.N; i++) {
                    opD->zD.vals[i] = 0.0;
                }
                for (int i = 0; i < opQ->zQ.N; i++) {
                    opQ->zQ.vals[i] = 0.0;
                }
                for (int i = 0; i < opE->zE.N; i++) {
                    opE->zE.vals[i] = 0.0;
                }
            }

            obj0 = obj1;
        }
    
        count++;
        ddebug[0] = count;
        fflush(stdout);
    }
    
    ddebug[13] = cvxop_gradient_getbval(opG, G);

    free(xbar.vals);
    free(taumx.vals);
    free(txmx.vals);
    free(tau.vals);
    free(bval_backlog);

    int bad_slew = cvxop_slewrate_check(opD, G);
    int bad_moments = cvxop_moments_check(opQ, G);
    int bad_gradient = cvxop_gradient_check(opG, G);

    ddebug[7] = bad_slew;
    ddebug[8] = bad_moments;
    ddebug[9] = bad_gradient;
}

void interp(cvx_mat *G, double dt_in, double dt_out, double TE, double T_readout) {
    int N1 = round((TE-T_readout) * 1.0e-3/dt_out);

    double *new_vals;
    double *temp_free = G->vals;
    new_vals = malloc(N1 * sizeof(double));


    double tt;
    double ti;
    int i0, i1;
    double d0, d1;
    double v0, v1;
    for (int i = 0; i < N1; i++) {
        ti = (dt_out * i) / dt_in;
        
        i0 = floor(ti);
        if (i0 < 0) {i0 = 0;} // Shouldn't happen unless some weird rounding and floor?
        
        i1 = i0+1;

        if (i1 < G->N) {
            d0 = fabs(ti-i1);
            d1 = 1.0 - d0;

            v0 = d0 * temp_free[i0];
            v1 = d1 * temp_free[i1];

            new_vals[i] = v0 + v1;
        } else {
            d0 = fabs(ti-i1);
            v0 = d0 * temp_free[i0];
            new_vals[i] = v0;
        }
    }

    G->vals = new_vals;
    G->N = N1;
    G->rows = N1;
    free(temp_free);
}



void run_kernel_diff(double **G_out, int *N_out, double **ddebug, int verbose,
                            int N, double dt, double gmax, double smax, double TE, 
                            int N_moments, double *moments_params, double PNS_thresh,  
                            double T_readout, double T_90, double T_180, int diffmode,
                            double bval_weight, double slew_weight, double moments_weight, 
                            double bval_reduce,  double dt_out,
                            int N_eddy, double *eddy_params,
                            int is_Gin, double *G_in)
{
    double relax = 1.7;

    if (verbose > 0) {
        printf ("\nFirst pass, N = %d    dt = %.2e\n\n", N, dt);
        fflush(stdout);
    }

    // This is the old style, but I don't think its right for when inversion time doesn't line up with dt
    int ind_inv = round((N + T_readout/(dt*1.0e3))/2.0);
    int ind_end90 = floor(T_90*(1e-3/dt));
    int ind_start180 = ind_inv - floor(T_180*(1e-3/dt/2));
    int ind_end180 = ind_inv + floor(T_180*(1e-3/dt/2));
    
    /*
    // Calculate times of inversion and rf dead times
    double t_inv = (N*dt + 1e-3 * T_readout) / 2.0;
    double t_end90 = 1e-3 * T_90;
    double t_start180 = t_inv - T_180*1e-3/2.0;
    double t_stop180 = t_inv + T_180*1e-3/2.0;
    
    // Get indices from times, always rounding in the most conservative direction
    int ind_inv = round(t_inv/dt);
    int ind_end90 = ceil(t_end90/dt);
    int ind_start180 = floor(t_start180/dt);
    int ind_end180 = ceil(t_stop180/dt);
    */

    if (verbose > 0) {
        printf ("\nN = %d  ind_inv = %d\n90_zeros = %d:%d    180_zeros = %d:%d\n\n", N, ind_inv, 0, ind_end90, ind_start180, ind_end180);
    }

    cvxop_beta opC;
    cvxop_bval opB;
    
    int N_converge = 1;
    double stop_increase = 1;

    if (diffmode == 1) {
        opB.active = 0; 
        opC.active = 1;
        N_converge = 24; 
        stop_increase = 1.0e-4;
    } else if (diffmode == 2) {
        opC.active = 0; 
        opB.active = 1;
        N_converge = 8;
        stop_increase = 1.0e-3; 
    }

    cvxop_beta_init(&opC, N, dt, bval_weight, verbose);
    cvxop_bval_init(&opB, N, ind_inv, dt, bval_weight, verbose);

    
    cvxop_gradient opG;
    cvxop_gradient_init(&opG, N, dt, gmax, ind_inv, verbose);
    cvxop_gradient_setFixRange(&opG, 0, ind_end90, 0.0);
    cvxop_gradient_setFixRange(&opG, ind_start180, ind_end180, 0.0);

    cvxop_slewrate opD;
    cvxop_slewrate_init(&opD, N, dt, smax, slew_weight, verbose);

    cvxop_pns opP;
    cvxop_pns_init(&opP, N, dt, ind_inv, PNS_thresh, verbose);

    cvxop_moments opQ;
    cvxop_moments_init(&opQ, N, ind_inv, dt, moments_weight, verbose);
    for (int i = 0; i < N_moments; i++) {
        cvxop_moments_addrow(&opQ, moments_params[MOMENTS_PARAMS_LEN*i+1], 
                                   moments_params[MOMENTS_PARAMS_LEN*i+4], 
                                   moments_params[MOMENTS_PARAMS_LEN*i+5]);
    }
    cvxop_moments_finishinit(&opQ);
    
    
    cvxop_eddy opE;
    cvxop_eddy_init(&opE, N, ind_inv, dt, .01, verbose);
    for (int i = 0; i < N_eddy; i++) {
        cvxop_eddy_addrow(&opE, (eddy_params[EDDY_PARAMS_LEN*i] * 1.0e-3), eddy_params[EDDY_PARAMS_LEN*i+1]);
    }
    cvxop_eddy_finishinit(&opE);
    
    cvx_mat G;
    
    cvxmat_alloc(&G, N, 1);
    cvxmat_setvals(&G, 0.0);
    if (is_Gin == 0) {   
        cvxop_init_G(&opG, &G);
    } else {
        for (int i = 0; i < N; i++) {
            G.vals[i] = G_in[i];
        }
    }

	*ddebug = (double *)malloc(480000*sizeof(double));
    for (int i = 0; i < 48; i++) {
        (*ddebug)[i] = 0.0;
    }

    cvx_optimize_kernel(&G, &opG, &opD, &opQ, &opE, &opC, &opB, &opP, N, relax, verbose, bval_reduce, *ddebug, N_converge, stop_increase);

    cvxop_gradient_limiter(&opG, &G);
    
    if (verbose > 0) {
        printf ("\n****************************************\n");
        printf ("--- Finished diff kernel4 #1 in %d iterations  bvalue = %.2f", (int)(*ddebug)[0], (*ddebug)[13]);
        printf ("\n****************************************\n");
    }

    if (dt_out > 0) {
        interp(&G, dt, dt_out, TE, T_readout);
    }

    *N_out = G.rows;
    *G_out = G.vals;

    (*ddebug)[2] = opB.weight;
    (*ddebug)[3]  = opQ.weight;
    (*ddebug)[4]  = opD.weight;

    (*ddebug)[10] = opQ.norms.vals[0];
    (*ddebug)[11] = opQ.norms.vals[1];
    (*ddebug)[12] = opQ.norms.vals[2];

    cvxop_gradient_destroy(&opG);
    cvxop_slewrate_destroy(&opD);
    cvxop_moments_destroy(&opQ);
    cvxop_beta_destroy(&opC);
    cvxop_bval_destroy(&opB);

    fflush(stdout);

}


void run_kernel_diff_fixedN(double **G_out, int *N_out, double **ddebug, int verbose,
                            int N0, double gmax, double smax, double TE, 
                            int N_moments, double *moments_params, double PNS_thresh,  
                            double T_readout, double T_90, double T_180, int diffmode, double dt_out,
                            int N_eddy, double *eddy_params)
{
    int N = N0;
    double dt = (TE-T_readout) * 1.0e-3 / (double) N;

    run_kernel_diff(G_out, N_out, ddebug, verbose, 
                        N, dt, gmax, smax, TE, 
                        N_moments, moments_params, PNS_thresh,
                        T_readout, T_90, T_180, diffmode,
                        10.0, 1.0, 10.0, 
                        10.0,  dt_out,
                        N_eddy, eddy_params,
                        0, NULL);
}


void run_kernel_diff_fixedN_Gin(double **G_out, int *N_out, double **ddebug, int verbose,
                                double *G_in, int N0, double gmax, double smax, double TE, 
                                int N_moments, double *moments_params, double PNS_thresh,  
                                double T_readout, double T_90, double T_180, int diffmode, double dt_out,
                                int N_eddy, double *eddy_params)
{
    int N = N0;
    double dt = (TE-T_readout) * 1.0e-3 / (double) N;

    run_kernel_diff(G_out, N_out, ddebug, verbose,
                        N, dt, gmax, smax, TE, 
                        N_moments, moments_params, PNS_thresh,
                        T_readout, T_90, T_180, diffmode,
                        10.0, 1.0, 10.0, 
                        10.0,  dt_out,
                        N_eddy, eddy_params,
                        1, G_in);
}




void run_kernel_diff_fixeddt(double **G_out, int *N_out, double **ddebug, int verbose,
                            double dt0, double gmax, double smax, double TE,
                            int N_moments, double *moments_params, double PNS_thresh, 
                            double T_readout, double T_90, double T_180, int diffmode, double dt_out,
                            int N_eddy, double *eddy_params)
{
    int N = round((TE-T_readout) * 1.0e-3/dt0);
    if (N < 5) {
        printf ("\nWARNING: N = %d looks too small, setting to 5\n\n", N);
        N = 5;
    }

    // double dt = (TE-T_readout) * 1.0e-3 / (double) N;
    double dt = dt0;

    run_kernel_diff(G_out, N_out, ddebug, verbose,
                            N, dt, gmax, smax, TE, 
                            N_moments, moments_params, PNS_thresh,
                            T_readout, T_90, T_180, diffmode,
                            10.0, 1.0, 10.0, 
                            10.0,  dt_out,
                            N_eddy, eddy_params,
                            0, NULL);

}


int main (void)
{
    printf ("In optimize_kernel.c main function\n");
    

    // 1 = betamax
    // 2 = bval max
    int diffmode = 2;

    double *G;
    int N;
    double *debug;

    int N_eddy = 0; 
    double *eddy_params;

    int N_moments = 0; 
    double *moments_params;

    double PNS_thresh = 0.0;

    double m_tol[3]={0.0, 0.0, 0.0};

    run_kernel_diff_fixedN(&G, &N, &debug, 1, 256, 0.04, 20.0, 200.0, N_moments, moments_params, PNS_thresh, 
                            12.0, 4.0, 8.0, diffmode, -1.0, N_eddy, eddy_params);

    return 0;
}
