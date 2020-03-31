#include <iostream> 
#include <string>
#include <math.h>  
#include <Eigen/Dense>

using namespace Eigen;
using namespace std; 

#include "op_moments.h"

Op_Moments::Op_Moments(int N, double dt, int Nc) 
    : Operator(N, dt, Nc, Nc)
{
    name = "Moments";
    moment_params.setZero(Nc, 7);
    A.setZero(Nc, N);
    mod.setZero(Nc);
    do_rw = true;     
}

void Op_Moments::prep_A()
{
    A.setZero();
    spec_norm2.setZero();
    
    double ref0 = 0.0; // This is used later to offset the waveform start time
    for(int i = 0; i < Nc; i++) {
        for(int j = 0; j < N; j++) {
            double order = moment_params(i, 1);
            double val = 1000.0 * 1000.0 * dt * pow( (1000.0 * (dt*j + ref0)), order );
            
            A(i, j) = val * inv_vec(j);
            spec_norm2(i) += val*val;
        }
    }
}

void Op_Moments::set_params(int N_moments, double* moment_params_in)
{
    if (Nc != N_moments) {
        cout << "ERROR: Nc not equal to N_moments in Op_Moments";
    }

    for(int i = 0; i < Nc; i++) {
        for(int j = 0; j < 7; j++) {
            moment_params(i, j) = moment_params_in[i*7 + j];
        }
        target(i) = moment_params(i, 5);
        tol0(i) = moment_params(i, 6);
        tol(i) = (1.0-cushion) * tol0(i);
        balance_mod(i) = 1.0 / tol(i);
    }

    prep_A();
}

void Op_Moments::set_params(MatrixXd &moment_params_in)
{
    if (Nc != moment_params_in.rows()) {
        cout << "ERROR: Nc not equal to moment_params.rows() in Op_Moments";
    }

    moment_params = moment_params_in;
    
    for(int i = 0; i < Nc; i++) {
        target(i) = moment_params(i, 5);
        tol0(i) = moment_params(i, 6);
        tol(i) = (1.0-cushion) * tol0(i);
        
        if (balanced) {
            balance_mod(i) = 1.0 / tol(i);
        } else {
            balance_mod(i) = 1.0;
        }
    }

    prep_A();
}

void Op_Moments::forward(VectorXd &X, VectorXd &out, 
                         bool apply_weight, int norm, bool no_balance)
{
    out = A*X;

    if (apply_weight) {
        out.array() *= weight.array();
    }

    if (balanced && !no_balance) {
        out.array() *= balance_mod.array();
    }

    if (norm == 2) {
        out.array() /= spec_norm2.array();
    }

}

void Op_Moments::transpose(VectorXd &X, VectorXd &out, 
                           bool apply_weight, int norm)
{   
    mod.setOnes();

    if (norm == 2) {
        mod.array() /= spec_norm2.array();
    }

    if (balanced) {
        mod.array() /= balance_mod.array();
    }

    if (apply_weight) {
        mod.array() *= weight.array();
    }

    // Just use mod as a temp solving array
    mod.array() *= X.array();
    out = A.transpose() * mod;
    out.array() *= fixer.array();

}


void Op_Moments::prox(VectorXd &X)
{
    for (int i = 0; i < X.size(); i++) {
        double lower_bound = balance_mod(i) * (target(i)-tol(i));
        double upper_bound = balance_mod(i) * (target(i)+tol(i));
        X(i) = X(i) < lower_bound ? lower_bound:X(i);
        X(i) = X(i) > upper_bound ? upper_bound:X(i);
    }
}