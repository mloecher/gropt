#include "op_pns.h"

/**
 * Initialize the opP struct
 * This is the operator that limits PNS based on the single exponential function found in the 
 * Schulte paper
 */
void cvxop_pns_init(cvxop_pns *opP, int N, double dt, int ind_inv, double thresh, int verbose) 
{
    opP->active = 0;
    opP->N = N;
    opP->dt = dt;
    opP->ind_inv = ind_inv; 
    opP->verbose = verbose;
    opP->thresh = thresh;

    cvxmat_alloc(&opP->coeff, N, 1);
    cvxmat_alloc(&opP->Ptau, N, 1);

    cvxmat_alloc(&opP->Px, N-1, 1);
    cvxmat_alloc(&opP->zP, N-1, 1);
    cvxmat_alloc(&opP->zPbuff, N-1, 1);
    cvxmat_alloc(&opP->zPbar, N-1, 1);
    

    double c = 334.0e-6;
    double Smin = 70.27027;
    for (int i = 0; i < N; i++) {
        opP->coeff.vals[i] = c / pow((c + dt*(N-1) - dt*i), 2.0) / Smin;
    }

}

/**
 * Keep this here for consistency, but these values are ~1e-14 so dont even bother
 */
void cvxop_pns_add2tau(cvxop_pns *opP, cvx_mat *tau_mat)
{
    // if (opP->active > 0) {
    //     for (int i = 0; i < opP->N; i++) {
    //         tau_mat->vals[i] += fabs(opB->sigBdenom.vals[i]);
    //     }
    // }

    return;
}

/**
 * Step the gradient waveform (taumx)
 */
void cvxop_pns_add2taumx(cvxop_pns *opP, cvx_mat *taumx)
{
    if (opP->active > 0) {
        
        // MATH: Ptau = P'*zP
        cvxmat_setvals(&(opP->Ptau), 0.0);

        for (int j = 0; j < (opP->N); j++) {
            for (int i = j; i < (opP->N); i++) {
                double val;
                if (i == 0) {
                    val = -opP->zP.vals[i];
                } else if (i == (opP->N-1)) {
                    val = opP->zP.vals[i-1];
                } else {
                    val = opP->zP.vals[i-1] - opP->zP.vals[i];
                }
                int c_ind = opP->N-1+j-i;
                opP->Ptau.vals[j] += opP->coeff.vals[c_ind] * val;
            }
        }

        // MATH: taumx -= Btau
        for (int i = 0; i < taumx->N; i++) {
            taumx->vals[i] += opP->Ptau.vals[i];
        }

    }
}

/**
 * Primal dual update
 */
void cvxop_pns_update(cvxop_pns *opP, cvx_mat *txmx, double rr)
{
    if (opP->active > 0) {

        // MATH: Px = (P*txmx)
        cvxmat_setvals(&(opP->Px), 0.0);
        for (int j = 0; j < (opP->N-1); j++) {
            for (int i = 0; i <= j; i++) {
                int c_ind = opP->N-1-j+i;
                opP->Px.vals[j] += opP->coeff.vals[c_ind] * (txmx->vals[i+1]-txmx->vals[i]);
            }
        }

        // MATH: Px = sigP*Px
        // Ignored for now, sigP is mostly 1

        // MATH: zPbuff  = = zP + Px = zP + sigP.*(P*txmx) 
        for (int i = 0; i < opP->zPbuff.N; i++) {
            opP->zPbuff.vals[i] = opP->zP.vals[i] + opP->Px.vals[i];
        }


        // MATH: zBbar = zBbuff - zBbar
        double cushion = 0.99;
        for (int i = 0; i < opP->zPbar.N; i++) {
            double val = opP->zPbuff.vals[i];
            if (val > cushion*opP->thresh) {
                opP->zPbar.vals[i] = cushion*opP->thresh;
            } else if (val < -cushion*opP->thresh) {
                opP->zPbar.vals[i] = -cushion*opP->thresh;
            } else {
                opP->zPbar.vals[i] = val;
            }
        }

        // MATH: zPbar = zPbuff - sigP*zPbar
        for (int i = 0; i < opP->zPbar.N; i++) {
            opP->zPbar.vals[i] = opP->zPbuff.vals[i] - opP->zPbar.vals[i];
        }

        // MATH: zP = rr*zPbar + (1-rr)*zP
        for (int i = 0; i < opP->zP.N; i++) {
            opP->zP.vals[i] = rr * opP->zPbar.vals[i] + (1 - rr) * opP->zP.vals[i];
        }

    }
}


int cvxop_pns_check(cvxop_pns *opP, cvx_mat *G)
{

    cvxmat_setvals(&(opP->Px), 0.0);
    for (int j = 0; j < (opP->N-1); j++) {
        for (int i = 0; i <= j; i++) {
            int c_ind = opP->N-1-j+i;
            opP->Px.vals[j] += opP->coeff.vals[c_ind] * (G->vals[i+1]-G->vals[i]);
        }
    }

    double max_pns = 0.0;
    for (int i = 0; i < (opP->N-1); i++) {
        if (fabs(opP->Px.vals[i]) > max_pns) {
            max_pns = fabs(opP->Px.vals[i]);
        }
    }
    
    printf("    Max PNS:     %f \n", max_pns);

    return 1;
}