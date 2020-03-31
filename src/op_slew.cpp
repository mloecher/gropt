#include <iostream> 
#include <string>
#include <math.h>  
#include <Eigen/Dense>

using namespace Eigen;
using namespace std; 

#include "op_slew.h"

Op_Slew::Op_Slew(int N, double dt) 
    : Operator(N, dt, 1, (N-1))
{
    name = "Slew";
    do_rw = true; 
}

void Op_Slew::set_params(double smax_in)
{

    target(0) = 0;
    tol0(0) = smax_in;
    tol(0) = (1.0-cushion) * tol0(0);

    if (balanced) {
        balance_mod(0) = 1.0 / tol(0);
    } else {
        balance_mod(0) = 1.0;
    }

    spec_norm2(0) = 4.0/dt/dt;
    smax = smax_in;

}

void Op_Slew::forward(VectorXd &X, VectorXd &out, 
                         bool apply_weight, int norm, bool no_balance)
{
    for (int i = 0; i < (N-1); i++) {
        out(i) = (X(i+1) - X(i))/dt;
    }

    if (apply_weight) {
        out.array() *= weight(0);
    }

    if (balanced && !no_balance) {
        out.array() *= balance_mod(0);
    }

    if (norm == 2) {
        out.array() /= spec_norm2(0);
    }

}

void Op_Slew::transpose(VectorXd &X, VectorXd &out, 
                           bool apply_weight, int norm)
{
    out(0) = -X(0) / dt;
    for (int i = 1; i < (N-1); i++) {
        out(i) = (X(i-1) - X(i)) / dt;
    }
    out(N-1) = X(N-2) / dt;

    if (norm == 2) {
        out.array() /= spec_norm2(0);
    }

    if (balanced) {
        out.array() /= balance_mod(0);
    }

    if (apply_weight) {
        out.array() *= weight(0);
    }

    out.array() *= fixer.array();

}


void Op_Slew::prox(VectorXd &X)
{
    for (int i = 0; i < X.size(); i++) {
        double lower_bound = balance_mod(0) * (target(0)-tol(0));
        double upper_bound = balance_mod(0) * (target(0)+tol(0));
        X(i) = X(i) < lower_bound ? lower_bound:X(i);
        X(i) = X(i) > upper_bound ? upper_bound:X(i);
    }
}