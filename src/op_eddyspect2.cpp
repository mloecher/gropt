#include <iostream> 
#include <string>
#include <math.h>  
#include "Eigen/Dense"

using namespace Eigen;
using namespace std; 

#include "op_eddyspect2.h"

Op_EddySpect2::Op_EddySpect2(int N, int Naxis, double dt, int Neddy_in, int i_start_in, int i_stop_in) 
    : Operator(N, Naxis, dt, 1, Neddy_in, false)
{
    name = "eddy spectrum2"; 
    do_rw = false;
    balance_mod(0) = 1.0;

    mode = E_MODE_END;
    max_lam = 60;  // 100 ms maximum lambda

    i_start = i_start_in;
    i_stop = i_stop_in;

    Neddy = Neddy_in;
    mod.setZero(Neddy);
    target_spect.setZero(Neddy);

    prep_A();
}

void Op_EddySpect2::set_spect(double *spect_in)
{
    for (int i = 0; i < Neddy; i++) {
        target_spect(i) = spect_in[i];
    }
}

void Op_EddySpect2::obj_add2b(VectorXd &b)
{
    // return;
    x_temp.setZero();
    transpose(target_spect, x_temp, true, 2);
    b += x_temp;
}


void Op_EddySpect2::prep_A()
{
    A.setZero(Neddy, Naxis*N);  // Do we want Naxis * Neddy rows?

    for (int i = 0; i < Neddy; i++) {
        double lam = (i * max_lam/(double)Neddy + 1.0e-4) * 1.0e-3;  // lambda in seconds
        
        for (int j = 0; j < N; j++) {
            double val = 0.0;
            double jj;

            if (j > i_stop) {continue;}

            jj = i_stop - j;
            if (jj >= 0) {
                val += exp(-jj*dt/lam);
            }

            jj = i_start - j;
            if (jj >= 0) {
                val -= exp(-jj*dt/lam);
            }

            A(i, j) = val;
        }

    }

    spec_norm2(0) = A.squaredNorm();
    
    // Eddy spectra in general seem to benefit from bigger values here.
    // TODO: Test with diffusion
    // spec_norm2(0) /= 1.e5;
}

void Op_EddySpect2::forward(VectorXd &X, VectorXd &out, 
                         bool apply_weight, int norm, bool no_balance)
{
    out = A*X;

    if (apply_weight) {
        // out -= target_spect;
        out.array() *= weight(0);
    }

    if (balanced && !no_balance) {
        out.array() *= balance_mod(0);
    }

    if (norm == 2) {
        out.array() /= spec_norm2(0);
    }
}

void Op_EddySpect2::transpose(VectorXd &X, VectorXd &out, 
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

void Op_EddySpect2::get_obj(VectorXd &X, int iiter)
{
    Ax_temp.setZero();
    forward(X, Ax_temp, false, 0, true);
    Ax_temp -= target_spect;
    current_obj = Ax_temp.squaredNorm();
    hist_obj(0, iiter) = current_obj;
}