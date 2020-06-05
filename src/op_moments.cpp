#include <iostream> 
#include <string>
#include <math.h>  
#include <Eigen/Dense>

using namespace Eigen;
using namespace std; 

#include "op_moments.h"

Op_Moments::Op_Moments(int N, int Naxis, double dt, int Nc) 
    : Operator(N, Naxis, dt, Nc, Nc, true)
{
    name = "Moments";
    moment_params.setZero(Nc, 7);
    A.setZero(Nc, Naxis*N);
    mod.setZero(Nc);
    do_rw = true;     
}

void Op_Moments::prep_A()
{
    A.setZero();
    spec_norm2.setZero();
    
    for(int i = 0; i < Nc; i++) {

        double axis = moment_params(i, 0);
        double order = moment_params(i, 1);
        double ref0 = moment_params(i, 2);
        double start = moment_params(i, 3);
        double stop = moment_params(i, 4);

        // Default start and stop is an entire axis
        int i_start = axis*N;
        int i_stop = (axis+1)*N;

        // Defined start and stop indices
        if (start > 0) {
            i_start = (int)(start+0.5) + axis*N;
        }
        if (stop > 0) {
            i_stop = (int)(stop+0.5) + axis*N;
        }

        for(int j = i_start; j < i_stop; j++) {
            double order = moment_params(i, 1);
            double jj = j - axis*N;
            double val = 1000.0 * 1000.0 * dt * pow( (1000.0 * (dt*jj + ref0)), order);
            
            A(i, j) = val * inv_vec(j);
            spec_norm2(i) += val*val;
        }

        spec_norm2(i) = sqrt(spec_norm2(i));
        // spec_norm2(i) = 1.0;
    }
}

// This is a special case where moments must be zero (for diffusion)
void Op_Moments::set_params_zeros(int N_moments, double moment_tol)
{
    if (Nc != N_moments) {
        cout << "ERROR: Nc not equal to N_moments in Op_Moments";
    }

    for(int i = 0; i < Nc; i++) {
        
        moment_params(i, 1) = (double)i;
        moment_params(i, 6) = moment_tol;

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
        
        if (balanced) {
            balance_mod(i) = 1.0 / tol(i);
        } else {
            balance_mod(i) = 1.0;
        }
    }

    prep_A();
}

void Op_Moments::set_params(MatrixXd &moment_params_in)
{
    if (Nc != moment_params_in.rows()) {
        cout << "ERROR: Nc not equal to moment_params.rows() in Op_Moments " << Nc << "  " << moment_params_in.rows() << endl;
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

// Overload because A needs to be recomputed
void Op_Moments::set_inv_vec(VectorXd &inv_vec_in)
{
    inv_vec = inv_vec_in;
    prep_A();
}