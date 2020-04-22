#include <iostream> 
#include <string>
#include <math.h>  
#include <Eigen/Dense>

using namespace Eigen;
using namespace std; 

#include "op_bval.h"

Op_BVal::Op_BVal(int N, double dt) 
    : Operator(N, dt, 1, N, false)
{
    name = "b-value"; 

    do_rw = false;
    balanced = false;

    GAMMA = 267.5221900e6;  // rad/S/T
    MAT_SCALE = pow((GAMMA / 1000.0 * dt), 2.0) * dt;  // 1/1000 is for m->mm in b-value

    spec_norm2(0) = (N*N+N)/2.0 * MAT_SCALE;
}

void Op_BVal::forward(VectorXd &X, VectorXd &out, 
                         bool apply_weight, int norm, bool no_balance)
{
    
    double gt = 0;    
    for (int i = 0; i < N; i++) {
        gt += X(i) * inv_vec(i);
        out(i) = gt * sqrt(MAT_SCALE);
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
    double gt = 0;    
    for (int i = N-1; i >= 0; i--) {
        gt += X(i) * sqrt(MAT_SCALE);
        out(i) = gt * inv_vec(i);
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
}

void Op_BVal::get_obj(VectorXd &X, int iiter)
{
    Ax_temp.setZero();
    forward(X, Ax_temp, false, 0, true);
    current_obj = Ax_temp.squaredNorm();
    hist_obj(0, iiter) = current_obj;
    
}