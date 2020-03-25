#ifndef CVX_OPMOMENTS_H
#define CVX_OPMOMENTS_H

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "cvx_matrix.h"

typedef struct {
    int active;
    int verbose;

    int N;
    int Naxis;
    int Ntotal;
    int Nrows;

    int ind_inv;
    double dt;

    double weight;

    cvx_mat Q0;
    cvx_mat Q;

    cvx_mat norms;
    cvx_mat weights;
    cvx_mat checks;
    cvx_mat tolerances;
    cvx_mat goals;


    cvx_mat sigQ;
    cvx_mat zQ;
    cvx_mat zQbuff;
    cvx_mat zQbar;
    cvx_mat Qx;


} cvxop_moments;


#define MAXROWS 16

void cvxop_moments_init(cvxop_moments *opQ, int N, int Naxis, int ind_inv, double dt,
                     double init_weight, int verbose);
void cvxop_moments_addrow(cvxop_moments *opQ, double *moments_params);
void cvxop_moments_finishinit(cvxop_moments *opE);

void compute_Qx(cvxop_moments *opQ, cvx_mat *txmx);
void compute_Qx2(cvxop_moments *opQ, cvx_mat *txmx);
void cvxop_moments_add2tau(cvxop_moments *opQ, cvx_mat *tau_mat);
void cvxop_moments_add2taumx(cvxop_moments *opQ, cvx_mat *taumx);
void cvxop_moments_update(cvxop_moments *opQ, cvx_mat *txmx, double relax);
int cvxop_moments_check(cvxop_moments *opQ, cvx_mat *G);
void cvxop_moments_reweight(cvxop_moments *opQ, double weight_mod);
void cvxop_moments_destroy(cvxop_moments *opQ);

#endif /* CVX_OPMOMENTS_H */