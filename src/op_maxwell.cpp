#include <iostream> 
#include <string>
#include <math.h>  
#include "Eigen/Dense"

using namespace Eigen;
using namespace std; 

#include "op_maxwell.h"

Op_Maxwell::Op_Maxwell(int N, int Naxis, double dt) 
    : Operator(N, Naxis, dt, 1, Naxis*N, false)
{
    name = "Maxwell"; 
    do_rw = false;
    balanced = false;
    // set_vals.setZero(Ntot);
}

void Op_Maxwell::set_params(double tol_in, int ind0_in, int ind1_in, int ind2_in)
{

    ind0 = ind0_in;
    ind1 = ind1_in;
    ind2 = ind2_in;

    target(0) = 0;
    tol0(0) = tol_in;
    tol(0) = (1.0-cushion) * tol0(0);

    if (balanced) {
        balance_mod(0) = 1.0 / tol(0);
    } else {
        balance_mod(0) = 1.0;
    }

    spec_norm2(0) = 1.0;
    tolerance = tol_in;

    // set_vals.setOnes();
    // for (int j = 0; j < Naxis; j++) {
    //     set_vals((j*N)) = 0.0;
    //     set_vals((j*N) + (N-1)) = 0.0;
    // }
    // set_vals.array() *= -9999999.0;

}

void Op_Maxwell::set_params(double tol_in, VectorXd &set_vals_in, int ind0_in, int ind1_in, int ind2_in)
{

    ind0 = ind0_in;
    ind1 = ind1_in;
    ind2 = ind2_in;

    target(0) = 0;
    tol0(0) = tol_in;
    tol(0) = (1.0-cushion) * tol0(0);

    if (balanced) {
        balance_mod(0) = 1.0 / tol(0);
    } else {
        balance_mod(0) = 1.0;
    }

    spec_norm2(0) = 1.0;
    tolerance = tol_in;

    // set_vals = set_vals_in;
}

void Op_Maxwell::forward(VectorXd &X, VectorXd &out, 
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

void Op_Maxwell::transpose(VectorXd &X, VectorXd &out, 
                           bool apply_weight, int norm)
{
    out = X;

    if (balanced) {
        out.array() /= balance_mod(0);
    }

    if (apply_weight) {
        out.array() *= weight(0);
    }

    out.array() *= fixer.array();
}


void Op_Maxwell::prox(VectorXd &X)
{

    if (rot_variant) {

        double norm0 = X.segment(ind0, ind1-ind0).norm();
        double norm1 = X.segment(ind1, ind2-ind1).norm();

        double bound = balance_mod(0) * (target(0)+tol(0));

        // cout << "Maxwell prox:  " << norm0  << "  " << norm1 << "  " << bound << endl;

        if ( (norm0 - norm1) > bound ) {
            X.segment(ind0, ind1-ind0) *= (norm1+bound)/norm0;
        } else if ( (norm0 - norm1) < -bound ) {
            X.segment(ind1, ind2-ind1) *= (norm0+bound)/norm1;
        }

        // for (int i = 0; i < X.size(); i++) {
        //     if (set_vals(i) > -10000) {
        //         X(i) = set_vals(i) * balance_mod(0);
        //     }
        // }

        norm0 = X.segment(ind0, ind1-ind0).norm();
        norm1 = X.segment(ind1, ind2-ind1).norm();
        // cout << "Maxwell prox2:  " << norm0  << "  " << norm1 << "  " << bound << endl;

    } else {
        //TODO
    }
}


void Op_Maxwell::check(VectorXd &X, int iiter)
{
    double check = 0.0;
    double scale = 0.0;

    if (rot_variant) {
        
        double norm0 = X.segment(ind0, ind1-ind0).norm();
        double norm1 = X.segment(ind1, ind2-ind1).norm();

        double bound = balance_mod(0) * (target(0)+tol0(0));

        if ( (norm0 - norm1) > bound ) {
            check = 1.0;
            scale = (norm1+bound)/norm0;
        } else if ( (norm0 - norm1) < -bound ) {
            check = 1.0;
            scale = (norm0+bound)/norm1;
        }

        // cout << "Maxwell check:  " << norm0  << "  " << norm1 << "  " << bound << "  " << check << endl;
        // cout << "Maxwell check2:  " << ind0  << "  " << ind1 << "  " << ind2 << endl;
        // cout << "Maxwell check3:  " << scale <<  endl;

    } else {
        //TODO
    }

    hist_check(0, iiter) = check;

}