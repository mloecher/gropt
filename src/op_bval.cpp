#include <iostream> 
#include <string>
#include <math.h>  
#include <Eigen/Dense>

using namespace Eigen;
using namespace std; 

#include "op_bval.h"

Op_BVal::Op_BVal(int N, int Naxis, double dt) 
    : Operator(N, Naxis, dt, 1, Naxis*N, false)
{
    name = "b-value"; 

    do_rw = false;
    balanced = false;

    balance_mod(0) = 1.0;

    GAMMA = 267.5221900e6;  // rad/S/T
    MAT_SCALE = pow((GAMMA / 1000.0 * dt), 2.0) * dt;  // 1/1000 is for m->mm in b-value

    spec_norm2(0) = (N*N+N)/2.0 * MAT_SCALE;
}

void Op_BVal::set_params(double bval_in)
{
    do_rw = true;

    target(0) = bval_in;
    tol0(0) = 1.0e-1;
    tol(0) = (1.0-cushion) * tol0(0);

    if (balanced) {
        balance_mod(0) = 1.0 / tol(0);
    } else {
        balance_mod(0) = 1.0;
    }

    bval0 = bval_in;

}

void Op_BVal::check(VectorXd &X, int iiter)
{
    double bval_t = (X/balance_mod(0)).squaredNorm();    
    
    feas_check(0) = fabs(bval_t - target(0));

    if (iiter%20 == 0) {
        cout << "   ^^^ bval_t " << bval_t << "  " << feas_check(0) << "  " << target(0) << "  " << tol0(0) << "  " << balance_mod(0) << endl;
    }

    for (int i = 0; i < feas_check.size(); i++) {
        if (feas_check[i] > tol0[i]) {
            hist_check(i, iiter) = 1.0;
        } else {
            hist_check(i, iiter) = 0.0;
        }
    }
}

void Op_BVal::forward(VectorXd &X, VectorXd &out, 
                         bool apply_weight, int norm, bool no_balance)
{
    for (int j = 0; j < Naxis; j++) {
        int jN = j*N;
        double gt = 0;    
        for (int i = 0; i < N; i++) {
            gt += X(jN + i) * inv_vec(jN + i);
            out(jN + i) = gt * sqrt(MAT_SCALE);
        }
    }

    if (apply_weight) {
        out.array() *= weight(0);
    }

    if (balanced && !no_balance) {
        out.array() *= balance_mod(0);
    }
}


void Op_BVal::transpose(VectorXd &X, VectorXd &out, 
                           bool apply_weight, int norm)
{
    for (int j = 0; j < Naxis; j++) {
        int jN = j*N;
        double gt = 0;    
        for (int i = N-1; i >= 0; i--) {
            gt += X(jN + i) * sqrt(MAT_SCALE);
            out(jN + i) = gt * inv_vec(jN + i);
        }
    }

    if (balanced) {
        out.array() /= balance_mod(0);
    }

    if (apply_weight) {
        out.array() *= weight(0);
    }

    if (norm == 2) {
        out.array() /= spec_norm2(0);
    }

    out.array() *= fixer.array();
}


void Op_BVal::prox(VectorXd &X)
{
    double xnorm = X.norm();
    double min_val = balance_mod(0) * sqrt(target(0) - tol(0));
    double max_val = balance_mod(0) * sqrt(target(0) + tol(0));

    if (xnorm < min_val) {
        X *= (min_val/xnorm);
    } else if (xnorm > max_val) {
        X *= (max_val/xnorm);
    }
}

void Op_BVal::get_obj(VectorXd &X, int iiter)
{
    Ax_temp.setZero();
    forward(X, Ax_temp, false, 0, true);
    current_obj = Ax_temp.squaredNorm();
    hist_obj(0, iiter) = current_obj;
    
}