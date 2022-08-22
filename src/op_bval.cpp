#include <iostream> 
#include <string>
#include <math.h>  
#include "Eigen/Dense"

using namespace Eigen;
using namespace std; 

#include "op_bval.h"

Op_BVal::Op_BVal(int N, int Naxis, double dt) 
    : Operator(N, Naxis, dt, Naxis, Naxis*N, false)
{
    name = "b-value"; 

    do_rw = false;
    balanced = false;
    start_ind = 0;
    
    for (int j = 0; j < Naxis; j++) {
        balance_mod(j) = 1.0;
    }

    GAMMA = 267.5221900e6;  // rad/S/T
    MAT_SCALE = pow((GAMMA / 1000.0 * dt), 2.0) * dt;  // 1/1000 is for m->mm in b-value
    
    for (int j = 0; j < Naxis; j++) {
        spec_norm2(j) = (N*N+N)/2.0 * MAT_SCALE;
    }
    
}

void Op_BVal::set_start(int start_ind_in)
{
    start_ind = start_ind_in;
    int Nb = N - start_ind_in;
    for (int j = 0; j < Naxis; j++) {
        spec_norm2(j) = (Nb*Nb+Nb)/2.0 * MAT_SCALE;
    }
}


void Op_BVal::set_params(double bval_in)
{
    do_rw = true;

    for (int j = 0; j < Naxis; j++) {
        target(j) = bval_in;
        tol0(j) = 1.0e-1;
        tol(j) = (1.0-cushion) * tol0(j);

        if (balanced) {
            balance_mod(j) = 1.0 / tol(j);
        } else {
            balance_mod(j) = 1.0;
        }
    }

    bval0 = bval_in;

}

void Op_BVal::set_params(double bval_in0, double bval_in1, double bval_in2)
{
    do_rw = true;

    target(0) = bval_in0;
    target(1) = bval_in1;
    target(2) = bval_in2;

    for (int j = 0; j < Naxis; j++) {
        tol0(j) = 1.0e-1;
        tol(j) = (1.0-cushion) * tol0(j);

        if (balanced) {
            balance_mod(j) = 1.0 / tol(j);
        } else {
            balance_mod(j) = 1.0;
        }
    }

    bval0 = bval_in0;

}


void Op_BVal::get_feas(VectorXd &s, int iiter)
{
    feas_temp = s;
    prox(feas_temp);
    feas_temp = s - feas_temp;

    for (int j = 0; j < Naxis; j++) {
        r_feas(j) = feas_temp.segment(j*N, N).norm()/s.segment(j*N, N).norm();
    }
    hist_feas.col(iiter) = r_feas;
}


void Op_BVal::check(VectorXd &X, int iiter)
{
    
    for (int j = 0; j < Naxis; j++) {
        double bval_t = (X.segment(j*N, N)/balance_mod(j)).squaredNorm();    
        
        feas_check(j) = fabs(bval_t - target(j));

        if (iiter%20 == 0) {
            cout << "   ^^^ bval_t " << iiter << "  " << bval_t << "  " << feas_check(j) << "  " << target(j) << "  " << tol0(j) << "  " << balance_mod(j) << " " << r_feas(j) << endl;
        }
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
    out.setZero();
    for (int j = 0; j < Naxis; j++) {
        int jN = j*N;
        double gt = 0;    
        for (int i = start_ind; i < N; i++) {
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
    out.setZero();
    for (int j = 0; j < Naxis; j++) {
        int jN = j*N;
        double gt = 0;    
        for (int i = N-1; i >= start_ind; i--) {
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

    for (int j = 0; j < Naxis; j++) {
        double xnorm = X.segment(j*N, N).norm();
        double min_val = balance_mod(j) * sqrt(target(j) - tol(j));
        double max_val = balance_mod(j) * sqrt(target(j) + tol(j));

        if (xnorm < min_val) {
            X.segment(j*N, N) *= (min_val/xnorm);
        } else if (xnorm > max_val) {
            X.segment(j*N, N) *= (max_val/xnorm);
        }
    }

}

void Op_BVal::get_obj(VectorXd &X, int iiter)
{
    Ax_temp.setZero();
    forward(X, Ax_temp, false, 0, true);
    current_obj = Ax_temp.squaredNorm();
    hist_obj(0, iiter) = current_obj;
    
}