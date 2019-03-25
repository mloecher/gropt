#include "op_bval.h"

/**
 * Initialize the opB struct
 * This is the operator that maximizes b-value for diffusion purposes
 */
void cvxop_bval_init(cvxop_bval *opB, int N, int ind_inv, double dt, double init_weight, int verbose) {
    
    opB->N = N;
    opB->dt = dt;
    opB->ind_inv = ind_inv;
    opB->verbose = verbose;
    opB->weight = init_weight;
    
    cvxmat_alloc(&opB->sigBdenom, N, 1);
    cvxmat_alloc(&opB->sigB, N, 1);

    cvxmat_alloc(&opB->csx, N, 1);
    cvxmat_alloc(&opB->Btau, N, 1);
    cvxmat_alloc(&opB->C, N, 1);

    cvxmat_alloc(&opB->zB, N, 1);
    cvxmat_alloc(&opB->zBbuff, N, 1);
    cvxmat_alloc(&opB->zBbar, N, 1);
    cvxmat_alloc(&opB->Bx, N, 1);

    // CG stuff, remove eventually
    cvxmat_alloc(&opB->b, N, 1);
    cvxmat_alloc(&opB->x, N, 1);
    cvxmat_alloc(&opB->r, N, 1);
    cvxmat_alloc(&opB->p, N, 1);
    cvxmat_alloc(&opB->Ap, N, 1);

    if (opB->active > 0) {

            cvxmat_alloc(&opB->C, N, 1);

            double tt;
            for (int i = 0; i < N; i++) {
                tt = N-i;
                opB->C.vals[i] = tt*(tt+1)/2.0;
            }
            for (int i = 0; i < N; i++) {
                tt = N;
                opB->C.vals[i] /= tt;
            }

            // These are all derived from the matrix B, which isn't calculated anymore
            // TODO:  Add some matlab/python code to show this matrix
            double mat_norm = 0.0;
            for (int i = 0; i < N; i++) {
                mat_norm += (2.0*i + 1.0) * pow((double)(N-i), 2.0);
            }
            mat_norm = sqrt(mat_norm);

            opB->mat_norm = mat_norm;
            
            opB->weight /= mat_norm;

            cvxop_bval_updatesigma(opB);

    }
}


void cvxop_bval_updatesigma(cvxop_bval *opB)
{
    cvxmat_setvals(&(opB->sigBdenom), 0.0);
    
    // // L0 norm
    // for (int i = 0; i < opB->N; i++) {
    //     opB->sigBdenom.vals[i] = 1.0;
    // } 

    // L1 norm
    for (int i = 0; i < opB->N; i++) {
        opB->sigBdenom.vals[opB->N-1-i] = opB->weight * ((double)i + 1.0) * ((double)opB->N - ((double)i/2.0));
    }

    // // L2 norm
    // for (int j = 0; j < opB->N; j++) {
    //     for (int i = 0; i < opB->N; i++) {
    //         int top = opB->N - j;
    //         int val = opB->N - i;
    //         if (val > top) {val = top;}
    //         double vald = val;
    //         opB->sigBdenom.vals[j] += (opB->weight * opB->weight * vald * vald);
    //     }
    //     opB->sigBdenom.vals[j] = sqrt(opB->sigBdenom.vals[j]);
    // }  

    // // Linf norm
    // for (int i = 0; i < opB->N; i++) {
    //     opB->sigBdenom.vals[i] = opB->weight * (double)(opB->N-i);
    // } 

    for (int i = 0; i < opB->N; i++) {
        opB->sigB.vals[i] = 1.0/opB->sigBdenom.vals[i];
    }
}



/**
 * Reweight the constraint and update all the subsequent weightings, and also the current descent direction zB
 * basically weight_mod * B
 */
void cvxop_bval_reweight(cvxop_bval *opB, double weight_mod)
{
    opB->weight *= weight_mod;

    cvxop_bval_updatesigma(opB);

    for (int i = 0; i < opB->zB.N; i++) {
        opB->zB.vals[i] *= weight_mod;
    }
    
}

/**
 * Add sigBdenom to the tau matrix 
 */
void cvxop_bval_add2tau(cvxop_bval *opB, cvx_mat *tau_mat)
{
    if (opB->active > 0) {
        for (int i = 0; i < opB->N; i++) {
            tau_mat->vals[i] += fabs(opB->sigBdenom.vals[i]);
        }
    }
}


/**
 * This replaces the old matrix multiplication, O(2N) instead of O(N^2)
 * out = B*in
 */
void cvxmat_bval_multB(cvx_mat *out, cvxop_bval *opB, cvx_mat *in) {
    
    // Integration
    double gt = 0.0;
    for (int i = 0; i < opB->N; i++) {
        if (i < opB->ind_inv) {
            gt -= in->vals[i];
        } else {
            gt += in->vals[i];
        }
        opB->csx.vals[i] = gt;
    }  

    // Integration transpose
    gt = 0.0;
    for (int i = (opB->N-1); i >= 0; i--) {
        gt += opB->csx.vals[i];
        if (i < opB->ind_inv) {
            out->vals[i] = gt;
        } else {
            out->vals[i] = -gt;
        }
    }

    // Apply weight
    for (int i = 0; i < out->N; i++) {
        out->vals[i] *= opB->weight;
    }
}

/**
 * Step the gradient waveform (taumx)
 */
void cvxop_bval_add2taumx(cvxop_bval *opB, cvx_mat *taumx)
{
    if (opB->active > 0) {
        
        // MATH: Btau = B*zB
        cvxmat_setvals(&(opB->Btau), 0.0);
        cvxmat_bval_multB(&opB->Btau, opB, &opB->zB);

        // MATH: taumx -= Btau
        for (int i = 0; i < taumx->N; i++) {
            taumx->vals[i] += -opB->Btau.vals[i];
        }

    }
}


/**
 * Primal dual update
 */
void cvxop_bval_update(cvxop_bval *opB, cvx_mat *txmx, double rr, double *ddebug, int count)
{
    if (opB->active > 0) {

        // MATH: Bx = (B*txmx)
        cvxmat_setvals(&(opB->Bx), 0.0);
        cvxmat_bval_multB(&opB->Bx, opB, txmx);

        // MATH: Bx = sigB*Bx
        for (int i = 0; i < opB->Bx.N; i++) {
            opB->Bx.vals[i] *= opB->sigB.vals[i];
        }

        // MATH: zBbuff  = = zB + Bx = zB + sigB.*(B*txmx) 
        for (int i = 0; i < opB->zBbuff.N; i++) {
            opB->zBbuff.vals[i] = opB->zB.vals[i] + opB->Bx.vals[i];
        }

        // MATH: zBbar = zBbuff - zBbar
        for (int i = 0; i < opB->zBbar.N; i++) {
            opB->zBbar.vals[i] = opB->zBbuff.vals[i];
        }

        // MATH: zE = rr*zEbar + (1-rr)*zE
        for (int i = 0; i < opB->zBbar.N; i++) {
            opB->zB.vals[i] = rr * opB->zBbar.vals[i] + (1 - rr) * opB->zB.vals[i];
        }
    }
}


void cvxop_bval_destroy(cvxop_bval *opB)
{
    free(opB->sigBdenom.vals);
    free(opB->sigB.vals);

    free(opB->csx.vals);
    free(opB->Btau.vals);
    free(opB->C.vals);
    
    free(opB->zB.vals);
    free(opB->zBbuff.vals);
    free(opB->zBbar.vals);
    free(opB->Bx.vals);

    free(opB->b.vals);
    free(opB->x.vals);
    free(opB->r.vals);
    free(opB->p.vals);
    free(opB->Ap.vals);
}






/**
 * Promximal mapping for the b-value operator
 */
void cvxop_bval_prox(cvxop_bval *opB)
{   
    double eps = 10.0;
    // b is zBbuff/sigB
    for (int i = 0; i < opB->r.N; i++) {
        opB->b.vals[i] = opB->zBbuff.vals[i];
        // opB->b.vals[i] = opB->zBbuff.vals[i]/opB->sigB.vals[i];
    }

    // Ap stores Ax to start with
    cvxmat_setvals(&(opB->Ap), 0.0);
    cvxmat_bval_multB(&opB->Ap, opB, &opB->x); 
    for (int i = 0; i < opB->r.N; i++) {
        // opB->Ap.vals[i] = opB->Ap.vals[i] / opB->sigB.vals[i] + opB->x.vals[i];
        opB->Ap.vals[i] = opB->Ap.vals[i] + eps * opB->x.vals[i];
    }

    // r = b-ax
    // p = r
    // rsold = r.T*r
    double rsnew = 0.0;
    double rsold = 0.0;

    for (int i = 0; i < opB->r.N; i++) {
        opB->r.vals[i] = opB->b.vals[i] - opB->Ap.vals[i];
        opB->p.vals[i] = opB->r.vals[i];
        rsold += opB->r.vals[i]*opB->r.vals[i];
    }

    for (int iter = 0; iter < 500; iter++) {
        // Ap
        cvxmat_setvals(&(opB->Ap), 0.0);
        cvxmat_bval_multB(&opB->Ap, opB, &opB->p); 
        for (int i = 0; i < opB->r.N; i++) {
            // opB->Ap.vals[i] = opB->Ap.vals[i] / opB->sigB.vals[i] + opB->p.vals[i];
            opB->Ap.vals[i] = opB->Ap.vals[i] + eps * opB->p.vals[i];
        }

        //p.T * Ap
        double pTAp = 0.0;
        for (int i = 0; i < opB->r.N; i++) {
            pTAp += opB->p.vals[i]*opB->Ap.vals[i];
        }

        double alpha = rsold / pTAp;

        // x = x + alpha * p
        // r = r - alpha * Ap
        // rsnew = r.T * r
        rsnew = 0.0;
        for (int i = 0; i < opB->r.N; i++) {
            opB->x.vals[i] = opB->x.vals[i] + alpha * opB->p.vals[i];
            opB->r.vals[i] = opB->r.vals[i] - alpha * opB->Ap.vals[i];
            rsnew += opB->r.vals[i]*opB->r.vals[i];
        }

        if (rsnew < 1.0e-20) {break;}

        for (int i = 0; i < opB->r.N; i++) {
            opB->p.vals[i] = opB->r.vals[i] + (rsnew / rsold) * opB->p.vals[i];
        }
        rsold = rsnew;
    }


}



/**
 * Promximal mapping for the b-value operator
 */
void cvxop_bval_proxxbar(cvxop_bval *opB, cvx_mat *xbar, double eps)
{   
    // b is -Bxbar
    cvxmat_setvals(&(opB->Bx), 0.0);
    cvxmat_bval_multB(&opB->Bx, opB, xbar);

    for (int i = 0; i < opB->r.N; i++) {
        opB->b.vals[i] = -opB->Bx.vals[i];
    }

    // Ap stores Ax to start with
    cvxmat_setvals(&(opB->Ap), 0.0);
    cvxmat_bval_multB(&opB->Ap, opB, &opB->x); 
    for (int i = 0; i < opB->r.N; i++) {
        opB->Ap.vals[i] = opB->Ap.vals[i] + eps * opB->x.vals[i];
    }

    // r = b-ax
    // p = r
    // rsold = r.T*r
    double rsnew = 0.0;
    double rsold = 0.0;

    for (int i = 0; i < opB->r.N; i++) {
        opB->r.vals[i] = opB->b.vals[i] - opB->Ap.vals[i];
        opB->p.vals[i] = opB->r.vals[i];
        rsold += opB->r.vals[i]*opB->r.vals[i];
    }

    for (int iter = 0; iter < 100; iter++) {
        // Ap
        cvxmat_setvals(&(opB->Ap), 0.0);
        cvxmat_bval_multB(&opB->Ap, opB, &opB->p); 
        for (int i = 0; i < opB->r.N; i++) {
            opB->Ap.vals[i] = opB->Ap.vals[i] + eps * opB->p.vals[i];
        }

        //p.T * Ap
        double pTAp = 0.0;
        for (int i = 0; i < opB->r.N; i++) {
            pTAp += opB->p.vals[i]*opB->Ap.vals[i];
        }

        double alpha = rsold / pTAp;

        // x = x + alpha * p
        // r = r - alpha * Ap
        // rsnew = r.T * r
        rsnew = 0.0;
        for (int i = 0; i < opB->r.N; i++) {
            opB->x.vals[i] = opB->x.vals[i] + alpha * opB->p.vals[i];
            opB->r.vals[i] = opB->r.vals[i] - alpha * opB->Ap.vals[i];
            rsnew += opB->r.vals[i]*opB->r.vals[i];
        }

        if (rsnew < 1.0e-12) {break;}

        for (int i = 0; i < opB->r.N; i++) {
            opB->p.vals[i] = opB->r.vals[i] + (rsnew / rsold) * opB->p.vals[i];
        }
        rsold = rsnew;
    }

    for (int i = 0; i < opB->r.N; i++) {
        xbar->vals[i] -= opB->x.vals[i];
    }


}
