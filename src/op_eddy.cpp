#include <iostream> 
#include <string>
#include <math.h>  
#include "Eigen/Dense" 

using namespace Eigen;
using namespace std; 

#include "op_eddy.h"

Op_Eddy::Op_Eddy(int N, int Naxis, double dt, int Nlam_in) 
    : Operator(N, Naxis, dt, Naxis*Nlam_in, Naxis*Nlam_in, true)
{
    name = "Eddy";
    Nlam = Nlam_in;
    Neddy = Naxis*Nlam;

    A.setZero(Neddy, N);
    mod.setZero(Neddy);

    do_rw = true;     
}

void Op_Eddy::prep_A(double lam, double tol_in)
{
    A.setZero(Neddy, N);

    for (int i = 0; i < Naxis; i++) {
        
        for (int j = 0; j < N; j++) {
            double jj = N - j - 1;
            double val = exp(-(jj+1.0)*dt/lam) - exp(-jj*dt/lam);
            A(i, j) = val;
        }

        target(i) = 0.0;
        tol0(i) = tol_in;
        tol(i) = (1.0-cushion) * tol0(i);
        
        if (balanced) {
            balance_mod(i) = 1.0 / tol(i);
        } else {
            balance_mod(i) = 1.0;
        }

        spec_norm2(i) = A.row(i).squaredNorm();

    }

}

void Op_Eddy::prep_A(VectorXd &lam_in, double tol_in)
{
    A.setZero(Neddy, N);

    for (int i = 0; i < Naxis; i++) {
        for (int i_lam = 0; i_lam < Nlam; i_lam++) {
            
            int ii = i_lam + i * Nlam;
            double lam = lam_in(i_lam);

            for (int j = 0; j < N; j++) {
                double jj = N - j - 1;
                double val = exp(-(jj+1.0)*dt/lam) - exp(-jj*dt/lam);
                A(ii, j) = val;
            }

            target(ii) = 0.0;
            tol0(ii) = tol_in;
            tol(ii) = (1.0-cushion) * tol0(ii);
            
            if (balanced) {
                balance_mod(ii) = 1.0 / tol(ii);
            } else {
                balance_mod(ii) = 1.0;
            }

            spec_norm2(ii) = A.row(ii).squaredNorm();

        }

    }
}

void Op_Eddy::prep_A(double lam, int eddy_stop, double tol_in)
{
    A.setZero(Neddy, N);

    for (int i = 0; i < Naxis; i++) {
        
        for (int j = 0; j < eddy_stop; j++) {
            double jj = eddy_stop - j - 1;
            double val = exp(-(jj+1.0)*dt/lam) - exp(-jj*dt/lam);
            A(i, j) = val;
        }

        target(i) = 0.0;
        tol0(i) = tol_in;
        tol(i) = (1.0-cushion) * tol0(i);
        
        if (balanced) {
            balance_mod(i) = 1.0 / tol(i);
        } else {
            balance_mod(i) = 1.0;
        }

        spec_norm2(i) = A.row(i).squaredNorm();
    }


}



void Op_Eddy::prep_A(double lam, int eddy_start, int eddy_stop, double tol_in)
{
    A.setZero(Neddy, N);

    for (int i = 0; i < Naxis; i++) {
        
        for (int j = eddy_start; j < eddy_stop; j++) {
            double jj = eddy_stop - j - 1;
            double val = exp(-(jj+1.0)*dt/lam) - exp(-jj*dt/lam);
            A(i, j) = val;
        }

        target(i) = 0.0;
        tol0(i) = tol_in;
        tol(i) = (1.0-cushion) * tol0(i);
        
        if (balanced) {
            balance_mod(i) = 1.0 / tol(i);
        } else {
            balance_mod(i) = 1.0;
        }

        spec_norm2(i) = A.row(i).squaredNorm();
    }


}


void Op_Eddy::prep_A(VectorXd &lam_in, int eddy_stop, double tol_in)
{
    A.setZero(Neddy, N);

    for (int i = 0; i < Naxis; i++) {
        for (int i_lam = 0; i_lam < Nlam; i_lam++) {
            
            int ii = i_lam + i * Nlam;
            double lam = lam_in(i_lam);
        
            for (int j = 0; j < eddy_stop; j++) {
                double jj = eddy_stop - j - 1;
                double val = exp(-(jj+1.0)*dt/lam) - exp(-jj*dt/lam);
                A(ii, j) = val;
            }

            target(ii) = 0.0;
            tol0(ii) = tol_in;
            tol(ii) = (1.0-cushion) * tol0(ii);
            
            if (balanced) {
                balance_mod(ii) = 1.0 / tol(ii);
            } else {
                balance_mod(ii) = 1.0;
            }

            spec_norm2(ii) = A.row(ii).squaredNorm();

        }
        
    }
}


void Op_Eddy::forward(VectorXd &X, VectorXd &out, 
                         bool apply_weight, int norm, bool no_balance)
{
    // TODO: Fix the Naxis Nlam selection
    for (int i = 0; i < Naxis; i++) {
        for (int i_lam = 0; i_lam < Nlam; i_lam++) {
            int ii = i_lam + i * Nlam;
            out(ii) = A.row(ii)*X.segment(i*N, N);
        }
    }

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

void Op_Eddy::transpose(VectorXd &X, VectorXd &out, 
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

    out.setZero();
    
    for (int i = 0; i < Naxis; i++) {
        for (int i_lam = 0; i_lam < Nlam; i_lam++) {
            int ii = i_lam + i * Nlam;
            out.segment(i*N, N) += A.row(ii).transpose()*mod(ii);
        }
    }

    out.array() *= fixer.array();

}


void Op_Eddy::prox(VectorXd &X)
{
    for (int i = 0; i < X.size(); i++) {
        double lower_bound = balance_mod(i) * (target(i)-tol(i));
        double upper_bound = balance_mod(i) * (target(i)+tol(i));
        X(i) = X(i) < lower_bound ? lower_bound:X(i);
        X(i) = X(i) > upper_bound ? upper_bound:X(i);
    }
}