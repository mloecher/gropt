#include <iostream> 
#include <string>
#include <math.h>  
#include "Eigen/Dense"

using namespace Eigen;
using namespace std; 

#include "op_eddyspect.h"

Op_EddySpect::Op_EddySpect(int N, int Naxis, double dt, int Neddy_in) 
    : Operator(N, Naxis, dt, 1, Neddy_in, false)
{
    name = "eddy spectrum"; 
    do_rw = false;
    balance_mod(0) = 1.0;

    mode = E_MODE_END;
    max_lam = 60;  // 100 ms maximum lambda

    Neddy = Neddy_in;
    mod.setZero(Neddy);
    prep_A();
}

void Op_EddySpect::prep_A()
{
    A.setZero(Neddy, Naxis*N);  // Do we want Naxis * Neddy rows?

    for (int i = 0; i < Neddy; i++) {
        double lam = (i * max_lam/(double)Neddy + 1.0e-4) * 1.0e-3;  // lambda in seconds
        for (int j = 0; j < N; j++) {
            double jj = N - j - 1;
            double val = exp(-(jj+1.0)*dt/lam) - exp(-jj*dt/lam);
            A(i, j) = val;
        }
    }

    spec_norm2(0) = A.squaredNorm();
    
    // Eddy spectra in general seem to benefit from bigger values here.
    // TODO: Test with diffusion
    // spec_norm2(0) /= 1.e5;
}

void Op_EddySpect::forward(VectorXd &X, VectorXd &out, 
                         bool apply_weight, int norm, bool no_balance)
{
    out = A*X;

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

void Op_EddySpect::transpose(VectorXd &X, VectorXd &out, 
                           bool apply_weight, int norm)
{   
    mod.setOnes();

    if (norm == 2) {
        mod.array() /= spec_norm2(0);
    }

    if (balanced) {
        mod.array() /= balance_mod(0);
    }

    if (apply_weight) {
        mod.array() *= weight(0);
    }

    // Just use mod as a temp solving array
    mod.array() *= X.array();
    out = A.transpose() * mod;
    out.array() *= fixer.array();

}

void Op_EddySpect::get_obj(VectorXd &X, int iiter)
{
    Ax_temp.setZero();
    forward(X, Ax_temp, false, 0, true);
    current_obj = Ax_temp.squaredNorm();
    hist_obj(0, iiter) = current_obj;
}