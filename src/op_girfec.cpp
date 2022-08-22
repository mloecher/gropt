#include <iostream> 
#include <string>
#include <math.h>  
#include "Eigen/Dense"

using namespace Eigen;
using namespace std; 

#include "op_girfec.h"

Op_GirfEC::Op_GirfEC(int N, int Naxis, double dt, VectorXcd &H_in, VectorXd &girf_win_in, int N_H) 
    : Operator(N, Naxis, dt, 1, N_H, false)  // For now N_H is N, but maybe this changes with padding or such
{
    name = "GirfEC"; 

    do_rw = false;
    balanced = false;

    N_FT = N_H;
    shape.push_back(N_FT);
    stride_d.push_back(sizeof(complex<double>));
    stride_cd.push_back(sizeof(complex<double>));
    axes.push_back(0);

    ft_c_vec0.setZero(N_FT);
    ft_c_vec1.setZero(N_FT);
    H = H_in;
    girf_win = girf_win_in;

    spec_norm2(0) = 1.0;
}


void Op_GirfEC::forward(VectorXd &X, VectorXd &out, 
                         bool apply_weight, int norm, bool no_balance)
{
    
    ft_c_vec0.setZero();
    ft_c_vec1.setZero();

    
    ft_c_vec0.real() = X;
    if (apply_weight) {
        ft_c_vec0.array() *= fixer.array();
    }

    pocketfft::c2c(shape, stride_d, stride_cd, axes, pocketfft::FORWARD,
                    ft_c_vec0.data(), ft_c_vec1.data(), 1.);


    // Multiply by H here
    ft_c_vec1.array() *= H.array();


    pocketfft::c2c(shape, stride_cd, stride_d, axes, pocketfft::BACKWARD,
                    ft_c_vec1.data(), ft_c_vec0.data(), 1.0/N_FT);


    out = ft_c_vec0.real();
    out.array() *= girf_win.array();


    if (apply_weight) {
        out.array() *= weight(0);
    }

    if (balanced && !no_balance) {
        out.array() *= balance_mod(0);
    }

}

void Op_GirfEC::transpose(VectorXd &X, VectorXd &out, 
                           bool apply_weight, int norm)
{
    
    ft_c_vec0.setZero();
    ft_c_vec1.setZero();

    ft_c_vec0.real() = X;

    pocketfft::c2c(shape, stride_d, stride_cd, axes, pocketfft::FORWARD,
                    ft_c_vec0.data(), ft_c_vec1.data(), 1.);


    ft_c_vec1.array() *= H.conjugate().array();

    pocketfft::c2c(shape, stride_cd, stride_d, axes, pocketfft::BACKWARD,
                    ft_c_vec1.data(), ft_c_vec0.data(), 1.0/N_FT);


    out = ft_c_vec0.real();

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

void Op_GirfEC::prox(VectorXd &X)
{
}

void Op_GirfEC::get_obj(VectorXd &X, int iiter)
{
    Ax_temp.setZero();
    forward(X, Ax_temp, false, 0, true);
    current_obj = Ax_temp.squaredNorm();
    hist_obj(0, iiter) = current_obj;
}