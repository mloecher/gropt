#include <iostream> 
#include <string>
#include <math.h>  
#include "Eigen/Dense"

using namespace Eigen;
using namespace std; 

#include "op_gradient.h"

Op_Gradient::Op_Gradient(int N, int Naxis, double dt) 
    : Operator(N, Naxis, dt, 1, Naxis*N, false)
{
    name = "Gradient"; 
    do_rw = false;
    balanced = false;
    set_vals.setZero(Ntot);
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
    for (int j = 0; j < Naxis; j++) {
        set_vals((j*N)) = 0.0;
        set_vals((j*N) + (N-1)) = 0.0;
    }
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


// void Op_Gradient::prox(VectorXd &X)
// {
//     for (int i = 0; i < X.size(); i++) {
//         double lower_bound = balance_mod(0) * (target(0)-tol(0));
//         double upper_bound = balance_mod(0) * (target(0)+tol(0));
//         X(i) = X(i) < lower_bound ? lower_bound:X(i);
//         X(i) = X(i) > upper_bound ? upper_bound:X(i);
//         if (set_vals(i) > -10000) {
//             X(i) = set_vals(i) * balance_mod(0);
//         }
//     }
// }



void Op_Gradient::prox(VectorXd &X)
{

    if (rot_variant) {
        for (int i = 0; i < X.size(); i++) {
            double lower_bound = balance_mod(0) * (target(0)-tol(0));
            double upper_bound = balance_mod(0) * (target(0)+tol(0));
            X(i) = X(i) < lower_bound ? lower_bound:X(i);
            X(i) = X(i) > upper_bound ? upper_bound:X(i);
            
            if (set_vals(i) > -10000) {
                X(i) = set_vals(i) * balance_mod(0);
            }
        }   
    } else {
        for (int i = 0; i < N; i++) {
            double upper_bound = balance_mod(0) * (target(0)+tol(0));
            
            double val = 0.0;
            for (int j = 0; j < Naxis; j++) {
                val += X(j*N+i)*X(j*N+i);
            }
            val = sqrt(val);

            if (val > upper_bound) {
                for (int j = 0; j < Naxis; j++) {
                    X(j*N+i) *= (upper_bound/val);
                }
            }

        }

        for (int i = 0; i < X.size(); i++) {
            if (set_vals(i) > -10000) {
                X(i) = set_vals(i) * balance_mod(0);
            }
        }
    }
}


void Op_Gradient::check(VectorXd &X, int iiter)
{
    double check = 0.0;

    if (rot_variant) {
        for (int i = 0; i < X.size(); i++) {
            double lower_bound = balance_mod(0) * (target(0)-tol0(0));
            double upper_bound = balance_mod(0) * (target(0)+tol0(0));

            if ((X(i) < lower_bound) || (X(i) > upper_bound)) {
                check = 1.0;
                // cout << "^^ Grad check fail at i = " << i << endl;
            }
        }   
    } else {
        for (int i = 0; i < N; i++) {
            double upper_bound = balance_mod(0) * (target(0)+tol0(0));
            
            double val = 0.0;
            for (int j = 0; j < Naxis; j++) {
                val += X(j*N+i)*X(j*N+i);
            }
            val = sqrt(val);

            if (val > upper_bound) {
                check = 1.0;
            }
        }
    }

    hist_check(0, iiter) = check;

}