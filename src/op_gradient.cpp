#include <iostream> 
#include <string>
#include <math.h>  
#include <Eigen/Dense>

using namespace Eigen;
using namespace std; 

#include "op_gradient.h"

Op_Gradient::Op_Gradient(int N, double dt) 
    : Operator(N, dt, 1, N)
{
    name = "Gradient"; 
    do_rw = false;
    set_vals.setZero(N);
}

void Op_Gradient::set_params(double gmax_in)
{

    target(0) = 0;
    tol0(0) = gmax_in;
    tol(0) = (1.0-cushion) * tol0(0);

    if (balanced) {
        balance_mod(0) = 1.0 / tol(0);
    } else {
        balance_mod(0) = 1.0;
    }

    spec_norm2(0) = 1.0;
    gmax = gmax_in;

    set_vals.setOnes();
    set_vals(0) = 0.0;
    set_vals(N-1) = 0.0;
    set_vals.array() *= -9999999.0;

}

void Op_Gradient::set_params(double gmax_in, VectorXd &set_vals_in)
{

    target(0) = 0;
    tol0(0) = gmax_in;
    tol(0) = (1.0-cushion) * tol0(0);

    if (balanced) {
        balance_mod(0) = 1.0 / tol(0);
    } else {
        balance_mod(0) = 1.0;
    }

    spec_norm2(0) = 1.0;
    gmax = gmax_in;

    set_vals = set_vals_in;
}

void Op_Gradient::forward(VectorXd &X, VectorXd &out, 
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

void Op_Gradient::transpose(VectorXd &X, VectorXd &out, 
                           bool apply_weight, int norm)
{
    out = X;

    if (balanced) {
        out.array() /= balance_mod(0);
    }

    if (apply_weight) {
        out.array() *= weight(0);
    }
}


void Op_Gradient::prox(VectorXd &X)
{
    for (int i = 0; i < X.size(); i++) {
        double lower_bound = balance_mod(0) * (target(0)-tol(0));
        double upper_bound = balance_mod(0) * (target(0)+tol(0));
        X(i) = X(i) < lower_bound ? lower_bound:X(i);
        X(i) = X(i) > upper_bound ? upper_bound:X(i);
        if (set_vals(i) > -10000) {
            X(i) = set_vals(i) * balance_mod(0);
        }
    }
}