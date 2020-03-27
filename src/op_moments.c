#include "op_moments.h"

/**
 * Initialize the opQ struct
 * This is the operator that matches eddy currents
 */
void cvxop_moments_init(cvxop_moments *opQ, int N, int Naxis, int ind_inv, double dt,
                     double init_weight, int verbose)
{
    opQ->active = 1;
    opQ->N = N;
    opQ->Naxis = Naxis;
    opQ->Ntotal = N * Naxis;
    opQ->ind_inv = ind_inv;
    opQ->dt = dt;
    opQ->verbose = verbose;

    opQ->Nrows = 0; // # of eddy current rows

    cvxmat_alloc(&opQ->Q0, MAXROWS, opQ->Ntotal);
    cvxmat_alloc(&opQ->Q, MAXROWS, opQ->Ntotal);

    cvxmat_alloc(&opQ->norms, MAXROWS, 1);
    cvxmat_alloc(&opQ->weights, MAXROWS, 1);
    cvxmat_alloc(&opQ->checks, MAXROWS, 1);
    cvxmat_alloc(&opQ->tolerances, MAXROWS, 1);
    cvxmat_alloc(&opQ->goals, MAXROWS, 1);
    cvxmat_alloc(&opQ->sigQ, MAXROWS, 1);

    cvxmat_alloc(&opQ->zQ, MAXROWS, 1);
    cvxmat_alloc(&opQ->zQbuff, MAXROWS, 1);
    cvxmat_alloc(&opQ->zQbar, MAXROWS, 1);
    cvxmat_alloc(&opQ->Qx, MAXROWS, 1);

    cvxmat_setvals(&opQ->weights, init_weight);
}


/**
 * Add a moment constraint
 * The *1000 x 3 are to get into units of mT/m*ms^X
 */
void cvxop_moments_addrow(cvxop_moments *opQ, double *moments_params)
{
    int axis = (moments_params[0] + 0.5);
    int order = (moments_params[1] + 0.5);
    double ref0 = moments_params[2];
    double start = moments_params[3];
    double stop = moments_params[4];
    double goal = moments_params[5];
    double tol = moments_params[6];

    int i_start = 0 + axis*opQ->N;
    int i_stop = (axis+1) * opQ->N;

    if (start > 0) {
        i_start = (int)(start+0.5) + axis*opQ->N;
    }
    if (stop > 0) {
        i_stop = (int)(stop+0.5) + axis*opQ->N;
    }

    for (int i = i_start; i < i_stop; i++) {
        double ii = i;
        double val = 1000.0 * 1000.0 * opQ->dt * pow( (1000.0 * (opQ->dt*ii + ref0)), (double)order );
        if (i > opQ->ind_inv) {val = -val;}
        cvxmat_set(&(opQ->Q0), opQ->Nrows, i, val);
    }

    opQ->tolerances.vals[opQ->Nrows] = tol;
    opQ->goals.vals[opQ->Nrows] = goal;
    opQ->Nrows += 1;
}


/**
 * Scale Q to have unit norm rows, and calculate sigQ
 */
void cvxop_moments_finishinit(cvxop_moments *opQ)
{
    // Calculate the row norms of the eddy current array and store
    for (int j = 0; j < opQ->Nrows; j++) {
        for (int i = 0; i < opQ->Ntotal; i++) {
            double temp = cvxmat_get(&(opQ->Q0), j, i);
            opQ->norms.vals[j] += temp*temp;
        }
        opQ->norms.vals[j] = sqrt(opQ->norms.vals[j]);
    }

    // Scale row norms to 1.0
    for (int j = 0; j < opQ->Nrows; j++) {
        opQ->weights.vals[j] /= opQ->norms.vals[j];
    }

    // Make Q as weighted Q0
    for (int j = 0; j < opQ->Nrows; j++) {
        for (int i = 0; i < opQ->Ntotal; i++) {
            double temp = cvxmat_get(&(opQ->Q0), j, i);
            cvxmat_set(&(opQ->Q), j, i, opQ->weights.vals[j] * temp);
        }
    }


    // Calculate sigQ as inverse of sum(abs(row of Q))
    for (int j = 0; j < opQ->Nrows; j++) {
        for (int i = 0; i < opQ->Ntotal; i++) {
            double temp = cvxmat_get(&(opQ->Q), j, i);
            opQ->sigQ.vals[j] += fabs(temp);
        }
        opQ->sigQ.vals[j] = 1.0 / opQ->sigQ.vals[j];
    }
}


/*
 * Reweight the constraint and update all the subsequent weightings, and also the current descent direction zQ
 * basically weight_mod * Q
 */
void cvxop_moments_reweight(cvxop_moments *opQ, double weight_mod)
{
    double ww;
    for (int j = 0; j < opQ->Nrows; j++) {
        ww = 1.0;
        if (opQ->checks.vals[j] > 0) {
            ww = weight_mod;
            // ww = 1.0 * opQ->checks.vals[j];
            // if (ww > weight_mod) {
            //     ww = weight_mod;
            // } else if (ww < 2.0) {
            //     ww = 2.0;
            // }
        }

        if (opQ->weights.vals[j] > 1.0e64) {
            ww = 1.0; // prevent overflow
        }

        opQ->weights.vals[j] *= ww;
        opQ->zQ.vals[j] *= ww;
    }

    // Make Q as weighted Q0
    for (int j = 0; j < opQ->Nrows; j++) {
        for (int i = 0; i < opQ->Ntotal; i++) {
            double temp = cvxmat_get(&(opQ->Q0), j, i);
            cvxmat_set(&(opQ->Q), j, i, opQ->weights.vals[j] * temp);
        }
    }

    // Calculate sigQ as inverse of sum(abs(row of Q))
    for (int j = 0; j < opQ->Nrows; j++) {
        for (int i = 0; i < opQ->Ntotal; i++) {
            double temp = cvxmat_get(&(opQ->Q), j, i);
            opQ->sigQ.vals[j] += fabs(temp);
        }
        opQ->sigQ.vals[j] = 1.0 / opQ->sigQ.vals[j];
    }
}


/**
 * Add absolute value of columns to the tau matrix 
 */
void cvxop_moments_add2tau(cvxop_moments *opQ, cvx_mat *tau_mat)
{
    for (int j = 0; j < opQ->Nrows; j++) {
        for (int i = 0; i < opQ->Ntotal; i++) {
            double temp = cvxmat_get(&(opQ->Q), j, i);
            tau_mat->vals[i] += fabs(temp);
        }
    }    
}



/**
 * Step the gradient waveform (taumx)
 */
void cvxop_moments_add2taumx(cvxop_moments *opQ, cvx_mat *taumx)
{   
    // MATH: taumx += E*zE
    for (int j = 0; j < opQ->Nrows; j++) {
        for (int i = 0; i < opQ->Ntotal; i++) {
            double temp = cvxmat_get(&(opQ->Q), j, i);
            taumx->vals[i] += (temp * opQ->zQ.vals[j]);
        }
    }

}


void compute_Qx(cvxop_moments *opQ, cvx_mat *txmx)
{
    int ii = 0;
    double temp;
    // MATH: Ex = E * txmx
    for (int j = 0; j < opQ->Nrows; j++) {
        temp = 0.0;
        for (int i = 0; i < opQ->Ntotal; i++) {
            // double temp = cvxmat_get(&(opQ->Q), j, i) * txmx->vals[i];
            // temp += opQ->Q.vals[ii] * txmx->vals[i];
            temp += opQ->Q.vals[j*opQ->Ntotal + i] * txmx->vals[i];
            ii++;
        }
        opQ->Qx.vals[j] = temp;
    }
}

void compute_Qx2(cvxop_moments *opQ, cvx_mat *txmx)
{
    double *Qpos = &(opQ->Q.vals[0]);
    double *Qxpos = &(opQ->Qx.vals[0]);
    double *xpos = &(txmx->vals[0]);
    double temp;
    // MATH: Ex = E * txmx
    for (int j = 0; j < opQ->Nrows; j++) {
        xpos = &(txmx->vals[0]);
        temp = 0.0;
        for (int i = 0; i < opQ->Ntotal; i++) {
            // double temp = cvxmat_get(&(opQ->Q), j, i) * txmx->vals[i];
            // temp += opQ->Q.vals[ii] * txmx->vals[i];
            temp += (*Qpos++) * (*xpos++);
        }
        *Qxpos = temp;
        *Qxpos++;
    }
}


/**
 * Primal dual update
 */
void cvxop_moments_update(cvxop_moments *opQ, cvx_mat *txmx, double rr)
{
    if (opQ->Nrows > 0) {

        compute_Qx2(opQ, txmx);

        double cushion = 0.99;

        // MATH: Ex = Ex * sigE
        for (int j = 0; j < opQ->Nrows; j++) {
            opQ->Qx.vals[j] *= opQ->sigQ.vals[j];
            opQ->zQbuff.vals[j] = opQ->zQ.vals[j] + opQ->Qx.vals[j];

            double low =  (opQ->goals.vals[j] - cushion*opQ->tolerances.vals[j]) * opQ->weights.vals[j];
            double high = (opQ->goals.vals[j] + cushion*opQ->tolerances.vals[j]) * opQ->weights.vals[j];
            double val = opQ->zQbuff.vals[j] / opQ->sigQ.vals[j];
            if (val < low) {
                opQ->zQbar.vals[j] = low;
            } else if (val > high) {
                opQ->zQbar.vals[j] = high;
            } else {
                opQ->zQbar.vals[j] = val;
            }

            opQ->zQbar.vals[j] = opQ->zQbuff.vals[j] - opQ->sigQ.vals[j] * opQ->zQbar.vals[j];

            opQ->zQ.vals[j] = rr * opQ->zQbar.vals[j] + (1 - rr) * opQ->zQ.vals[j];
        }

    }
}




/*
 * Check if moments are larger than a fixed tolerance
 */
int cvxop_moments_check(cvxop_moments *opQ, cvx_mat *G)
{
    cvxmat_setvals(&(opQ->Qx), 0.0);

    // MATH: Ex = E * txmx
    for (int j = 0; j < opQ->Nrows; j++) {
        for (int i = 0; i < opQ->Ntotal; i++) {
            double temp = cvxmat_get(&(opQ->Q0), j, i) * G->vals[i];
            opQ->Qx.vals[j] += temp;
        }
    }

    // Set checks to be 0 if within tolerance, otherwise set to the ratio of eddy current to tolerance
    cvxmat_setvals(&(opQ->checks), 0.0);
    for (int j = 0; j < opQ->Nrows; j++) {
        double tol = opQ->tolerances.vals[j];
        double low =  opQ->goals.vals[j] - tol;
        double high = opQ->goals.vals[j] + tol;
        double diff;
        if (opQ->Qx.vals[j] < low) {
            diff = opQ->goals.vals[j] - opQ->Qx.vals[j];
            opQ->checks.vals[j] = diff / tol;
        } else if (opQ->Qx.vals[j] > high) {
            diff = opQ->Qx.vals[j] - opQ->goals.vals[j];
            opQ->checks.vals[j] = diff / tol;
        }
    }


    int moments_bad = 0;

    for (int j = 0; j < opQ->Nrows; j++) {
        if (opQ->checks.vals[j] > 0) {
             moments_bad = 1;
        }
    }

    if (opQ->verbose>0) {   
        printf("    Moments check:  (%d)  [%.2e %.2e %.2e]  %.2e  %.2e  %.2e   %d \n", moments_bad,  
        opQ->weights.vals[0], opQ->weights.vals[1], opQ->weights.vals[2], opQ->Qx.vals[0], opQ->Qx.vals[1], opQ->Qx.vals[2],
        opQ->Nrows);
    }

    return moments_bad;
}


/*
 * Free memory
 */
void cvxop_moments_destroy(cvxop_moments *opQ)
{

    free(opQ->norms.vals);
    free(opQ->weights.vals);
    free(opQ->checks.vals);
    free(opQ->tolerances.vals);
    free(opQ->goals.vals);


    free(opQ->Q0.vals);
    free(opQ->Q.vals);

    free(opQ->sigQ.vals);

    free(opQ->zQ.vals);
    free(opQ->zQbuff.vals);
    free(opQ->zQbar.vals);
    free(opQ->Qx.vals);
}

