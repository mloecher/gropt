#include <iostream> 
#include <string>
#include <math.h>  
#include "Eigen/Dense"

using namespace Eigen;
using namespace std; 

#include "op_duty.h"

Op_Duty::Op_Duty(int N, int Naxis, double dt) 
    : Operator(N, Naxis, dt, 1, Naxis*N, false)
{
    name = "duty cycle"; 

    do_rw = false;
    balanced = false;

    spec_norm2(0) = 1.0;
}

void Op_Duty::forward(VectorXd &X, VectorXd &out, 
                         bool apply_weight, int norm, bool no_balance)
{
    
    out = X;

    if (apply_weight) {
        out.array() *= weight(0);
    }

    if (balanced && !no_balance) {
        out.array() *= balance_mod(0);
    }
}

void Op_Duty::transpose(VectorXd &X, VectorXd &out, 
                           bool apply_weight, int norm)
{
    out = X;

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

void Op_Duty::prox(VectorXd &X)
{
}

void Op_Duty::get_obj(VectorXd &X, int iiter)
{
    Ax_temp.setZero();
    forward(X, Ax_temp, false, 0, true);
    current_obj = Ax_temp.squaredNorm();
    hist_obj(0, iiter) = current_obj;
    
}