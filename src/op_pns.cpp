#include <iostream> 
#include <string>
#include <math.h>  
#include "Eigen/Dense"

using namespace Eigen;
using namespace std; 

#include "op_pns.h"

Op_PNS::Op_PNS(int N, int Naxis, double dt) 
    : Operator(N, Naxis, dt, 1, Naxis*(N-1), false)
{
    name = "PNS";
    do_rw = true; 

    coeff.setZero(N);
    double c = 334.0e-6;
    double Smin = 60;
    for (int i = 0; i < N; i++) {
        coeff(i) = c / pow((c + dt*(N-1) - dt*i), 2.0) / Smin;
    }
    
    spec_norm2(0) = coeff.squaredNorm();
}

void Op_PNS::set_params(double stim_thresh_in)
{

    target(0) = 0;
    tol0(0) = stim_thresh_in;
    tol(0) = (1.0-cushion) * tol0(0);

    if (balanced) {
        balance_mod(0) = 1.0 / tol(0);
    } else {
        balance_mod(0) = 1.0;
    }

    stim_thresh = stim_thresh_in;
}

void Op_PNS::forward(VectorXd &X, VectorXd &out, 
                         bool apply_weight, int norm, bool no_balance)
{
    
    out.setZero();
    
    for (int i_ax = 0; i_ax < Naxis; i_ax++) {

        for (int j = 0; j < (N-1); j++) {
            for (int i = 0; i <= j; i++) {
                int c_ind = N-1-j+i;
                out(i_ax*(N-1) + j) += coeff(c_ind) * (X(i_ax*N + i+1) - X(i_ax*N + i));
            }
        }

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

void Op_PNS::transpose(VectorXd &X, VectorXd &out, 
                           bool apply_weight, int norm)
{
    out.setZero();

    for (int i_ax = 0; i_ax < Naxis; i_ax++) {

        for (int j = 0; j < N; j++) {
            for (int i = j; i < N; i++) {
                int c_ind = N-1+j-i;

                if (i == 0) {
                    out(i_ax*N + j) += coeff(c_ind) * (-X(i_ax*(N-1) + i));
                } else if (i == (N-1)) {
                    out(i_ax*N + j) += coeff(c_ind) * (X(i_ax*(N-1) + i-1));
                } else {
                    out(i_ax*N + j) += coeff(c_ind) * (X(i_ax*(N-1) + i-1) - X(i_ax*(N-1) + i));
                }
                
            }
        }

    }
    

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


// void Op_Slew::prox(VectorXd &X)
// {
//     for (int i = 0; i < X.size(); i++) {
//         double lower_bound = balance_mod(0) * (target(0)-tol(0));
//         double upper_bound = balance_mod(0) * (target(0)+tol(0));
//         X(i) = X(i) < lower_bound ? lower_bound:X(i);
//         X(i) = X(i) > upper_bound ? upper_bound:X(i);
//     }
// }


void Op_PNS::prox(VectorXd &X)
{
    for (int i = 0; i < (N-1); i++) {
        double upper_bound = balance_mod(0) * (target(0)+tol(0));
        
        double val = 0.0;
        for (int j = 0; j < Naxis; j++) {
            val += X(j*(N-1)+i)*X(j*(N-1)+i);
        }
        val = sqrt(val);

        if (val > upper_bound) {
            for (int j = 0; j < Naxis; j++) {
                X(j*(N-1)+i) *= (upper_bound/val);
            }
        }
    }
}

void Op_PNS::check(VectorXd &X, int iiter)
{
    double check = 0.0;

    for (int i = 0; i < (N-1); i++) {
        double upper_bound = balance_mod(0) * (target(0)+tol(0));
        
        double val = 0.0;
        for (int j = 0; j < Naxis; j++) {
            val += X(j*(N-1)+i) * X(j*(N-1)+i);
        }
        val = sqrt(val);

        if (val > upper_bound) {
            check = 1.0;
        }
    }

    hist_check(0, iiter) = check;
}