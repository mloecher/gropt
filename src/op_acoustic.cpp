#include <iostream> 
#include <string>
#include <math.h>  
#include "Eigen/Dense"

using namespace Eigen;
using namespace std; 

#include "op_acoustic.h"

Op_Acoustic::Op_Acoustic(int N, int Naxis, double dt, VectorXcd &H_in) 
    : Operator(N, Naxis, dt, 1, 15002, false)
{
    name = "acoustic"; 

    do_rw = false;
    balanced = false;

    N_FT = 15002;
    shape.push_back(N_FT);
    stride_d.push_back(sizeof(double));
    stride_cd.push_back(sizeof(complex<double>));
    axes.push_back(0);

    ft_vec.setZero(N_FT);
    ft_c_vec.setZero(N_FT);
    H = H_in;

    spec_norm2(0) = 1.0;
}

Op_Acoustic::Op_Acoustic(int N, int Naxis, double dt, VectorXcd &H_in, int N_H) 
    : Operator(N, Naxis, dt, 1, 2*(N_H-1), false)
{
    name = "acoustic"; 

    do_rw = false;
    balanced = false;

    N_FT = 2*(N_H-1);
    shape.push_back(N_FT);
    stride_d.push_back(sizeof(double));
    stride_cd.push_back(sizeof(complex<double>));
    axes.push_back(0);

    ft_vec.setZero(N_FT);
    ft_c_vec.setZero(N_H);
    H = H_in;

    spec_norm2(0) = 1.0;
}

void Op_Acoustic::forward(VectorXd &X, VectorXd &out, 
                         bool apply_weight, int norm, bool no_balance)
{
    
    ft_vec.setZero();
    ft_c_vec.setZero();

    
    ft_vec.head(X.size()) = X;
    if (apply_weight) {
        ft_vec.head(X.size()).array() *= fixer.array();
    }
    
    /*
    int Nrep = 5;
    for (int i = 0; i < Nrep; i++) {
        ft_vec.segment(100+i*X.size(), X.size()) = X;
        // if (apply_weight) {
        //     ft_vec.segment(i*X.size(), X.size()).array() *= fixer.array();
        // }
    }
    */

    pocketfft::r2c(shape, stride_d, stride_cd, axes, pocketfft::FORWARD,
                    ft_vec.data(), ft_c_vec.data(), 1.);


    // Multiply by H here
    ft_c_vec.array() *= H.array();

    for (int i = 0; i < N_FT/2; i++) {
        out(2*i) = ft_c_vec(i).real();
        out(2*i+1) = ft_c_vec(i).imag();
    }

    if (apply_weight) {
        out.array() *= weight(0);
    }

    if (balanced && !no_balance) {
        out.array() *= balance_mod(0);
    }

}

void Op_Acoustic::transpose(VectorXd &X, VectorXd &out, 
                           bool apply_weight, int norm)
{
    
    ft_vec.setZero();
    ft_c_vec.setZero();

    for (int i = 0; i < N_FT/2; i++) {
        ft_c_vec(i) = complex<double>(X(2*i), X(2*i+1));
    }

    ft_c_vec.array() *= H.conjugate().array();

    pocketfft::c2r(shape, stride_cd, stride_d, axes, pocketfft::BACKWARD,
                    ft_c_vec.data(), ft_vec.data(), 1.0/N_FT);


    out = ft_vec.head(N);
    
    // int i_rep = 2;
    // out = ft_vec.segment(i_rep*out.size(), out.size());

    /*
    out.setZero();
    int Nrep = 5;
    for (int i = 0; i < Nrep; i++) {
        out.array() += ft_vec.segment(100+i*out.size(), out.size()).array();
    }
    out.array() /= Nrep;
    */

    if (balanced) {
        out.array() /= balance_mod(0);
    }

    if (apply_weight) {
        out.array() *= weight(0);
    }

    if (norm == 2) {
        out.array() /= spec_norm2(0);
    }

    out.array() *= fixer.array();

}

void Op_Acoustic::prox(VectorXd &X)
{
}

void Op_Acoustic::get_obj(VectorXd &X, int iiter)
{
    Ax_temp.setZero();
    forward(X, Ax_temp, false, 0, true);
    current_obj = Ax_temp.squaredNorm();
    hist_obj(0, iiter) = current_obj;
}