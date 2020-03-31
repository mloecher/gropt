#include <iostream> 
#include <string>
#include <math.h>  
#include <Eigen/Dense>

using namespace Eigen;
using namespace std; 

#include "op_main.h"

Operator::Operator(int N, double dt, int Nc, int Ax_size) 
    : N(N), dt(dt), Nc(Nc), Ax_size(Ax_size)
{
    name = "OperatorMain";

    weight.setOnes(Nc);
    gamma.setOnes(Nc);

    cushion = 1.0e-3;
    tol0.setOnes(Nc);
    tol.setOnes(Nc);
    target.setOnes(Nc);
    
    spec_norm2.setOnes(Nc);
    balanced = true;
    balance_mod.setOnes(Nc);

    x_temp.setZero(N);
    Ax_temp.setZero(Ax_size);

    Y0.setZero(Ax_size);
    Y1.setZero(Ax_size);
    U0.setZero(Ax_size);
    U1.setZero(Ax_size);

    s.setZero(Ax_size);
    xbar.setZero(Ax_size);

    Uhat00.setZero(Ax_size);
    U00.setZero(Ax_size);
    s00.setZero(Ax_size);
    Y00.setZero(Ax_size);

    r_feas.setZero(Nc);
    feas_temp.setZero(Ax_size);
    feas_check.setZero(Nc);

    e_corr = 0.5;
    do_rw = false;
    rw_eps = 1e-6;
    rw_scalelim = 2.0;
    rw_interval = 8;

    weight_min = 1.0e-4;
    weight_max = 1.0e64;

    fixer.setOnes(N);
    fixer(0) = 0.0;
    fixer(N-1) = 0.0;
    inv_vec.setOnes(N);

    allocate_rwvecs();
}

void Operator::allocate_rwvecs()
{
    int alloc_size;
    if (Nc == 1) {
        alloc_size = Ax_size;
    } else {
        alloc_size = 1;
    }
    
    uhat1.setZero(alloc_size);
    duhat.setZero(alloc_size);
    du.setZero(alloc_size);
    dhhat.setZero(alloc_size);
    dghat.setZero(alloc_size);
}


void Operator::reweight()
{
    for (int ii = 0; ii < Nc; ii++) {
        double rho0 = weight(ii);

        if (Nc == 1) {
            uhat1.array() = U0.array() + rho0*(Y0.array() - s.array());
            duhat.array() = uhat1.array() - Uhat00.array();
            du.array() = U1.array() - U00.array();
            dhhat.array() = s.array() - s00.array();
            dghat.array() = -(Y1.array() - Y00.array());
        } else {
            // These "vectors" are just 1 element doubles, but keep 
            // them vectors for compatibility with the rest of this function
            uhat1(0) = U0(ii) + rho0*(Y0(ii) - s(ii));
            duhat(0) = uhat1(0) - Uhat00(ii);
            du(0) = U1(ii) - U00(ii);
            dhhat(0) = s(ii) - s00(ii);
            dghat(0) = -(Y1(ii) - Y00(ii));
        }

        double norm_dhhat_duhat = dhhat.norm()*duhat.norm();
        double dot_dhhat_dhhat = dhhat.dot(dhhat);
        double dot_dhhat_duhat = dhhat.dot(duhat);

        
        double alpha_corr = 0.0;
        if ((norm_dhhat_duhat > rw_eps) 
            && (dot_dhhat_dhhat > rw_eps) 
            && (dot_dhhat_duhat > rw_eps)) {
                alpha_corr = dhhat.dot(duhat)/norm_dhhat_duhat;
            }

        // cout << "RW1: " << name << " " << norm_dhhat_duhat << "  --  " << dot_dhhat_dhhat << "  --  " << dot_dhhat_duhat << "  --  " << alpha_corr << endl;

        double norm_dghat_du = dghat.norm()*du.norm();
        double dot_dghat_dghat = dghat.dot(dghat);
        double dot_dghat_du = dghat.dot(du);

        double beta_corr = 0.0;
        if ((norm_dghat_du > rw_eps) 
            && (dot_dghat_dghat > rw_eps) 
            && (dot_dghat_du > rw_eps)) {
                beta_corr = dghat.dot(du)/norm_dghat_du;
            }

        double alpha = 0.0;
        if (alpha_corr > e_corr) {
            double alpha_mg = dhhat.dot(duhat)/dot_dhhat_dhhat;
            double alpha_sd = duhat.dot(duhat)/dot_dhhat_duhat;
            if (2.0*alpha_mg > alpha_sd) {
                alpha = alpha_mg;
            } else {
                alpha = alpha_sd - 0.5*alpha_mg;
            }
        }

        double beta = 0.0;
        if (beta_corr > e_corr) {
            double beta_mg = dghat.dot(du)/dot_dghat_dghat;
            double beta_sd = du.dot(du)/dot_dghat_du;
            if (2.0*beta_mg > beta_sd) {
                beta = beta_mg;
            } else {
                beta = beta_sd - 0.5*beta_mg;
            }
        }

        double step_g1 = 0.0;
        double gamma1 = 0.0;
        if (alpha_corr > e_corr) {
            if (beta_corr > e_corr) {
                step_g1 = sqrt(alpha*beta);
                gamma1 = 1.0 + 2.0*sqrt(alpha*beta)/(alpha+beta);
            } else {
                step_g1 = alpha;
                gamma1 = 1.9;
            }
        } else {
            if (beta_corr > e_corr) {
                step_g1 = beta;
                gamma1 = 1.1;
            } else {
                step_g1 = rho0;
                gamma1 = 1.5;
            }
        }


        if (step_g1 > rw_scalelim*weight(ii)) {
            weight(ii) *= rw_scalelim;
        } else if (rw_scalelim*step_g1 < weight(ii)) {
            weight(ii) *= 1.0/rw_scalelim;
        } else {   
            weight(ii) = step_g1;
        }

        gamma(ii) = gamma1;

        if (Nc == 1) {
            Uhat00 = uhat1;
            U00 = U1;
            s00 = s;
            Y00 = Y1;
        } else {
            Uhat00(ii) = uhat1(0);
            U00(ii) = U1(ii);
            s00(ii) = s(ii);
            Y00(ii) = Y1(ii);
        }
    }
}


void Operator::prep_y(VectorXd &X)
{   
    forward(X, Y0, false, 0, false);
    Y00 = Y0;
}

void Operator::forward(VectorXd &X, VectorXd &out, bool apply_weight, int norm, bool no_balance)
{
    out = X;
}

void Operator::transpose(VectorXd &X, VectorXd &out, bool apply_weight, int norm)
{
    out = X;
}

void Operator::prox(VectorXd &X)
{
}

void Operator::update(VectorXd &X, int iiter)
{
    forward(X, s, false, 0, false);

    // Is there a better way to handle the single element gamma?  
    // We could switch it back to doubles, but I don't like the type mixing.
    if (Nc == 1) {
        xbar = gamma(0) * s + (1.0-gamma(0))*Y0;
        Y1 = xbar - U0/weight(0);
    } else {
        xbar.array() = gamma.array() * s.array() + (1.0-gamma.array())*Y0.array();
        Y1.array() = xbar.array() - U0.array()/weight.array();
    }

    prox(Y1);

    if (Nc == 1) {
        U1.array() = U0 + weight(0)*(Y1-xbar);
    } else {
        U1.array() = U0.array() + weight.array()*(Y1.array()-xbar.array());
    }

    // Update feasibility metrics
    feas_temp = s;
    prox(feas_temp);
    feas_temp = s - feas_temp;
    if (Nc == 1) {
        r_feas(0) = feas_temp.norm()/s.norm();
    } else {
        r_feas.array() = feas_temp.array()/s.array();
    }

    // cout << "Reweighting " << name << "  --  " << weight.transpose() << "  --  " << gamma.transpose() <<endl;
    // Do reweighting of weight and gamma
    if ((do_rw) && (iiter > rw_interval) && (iiter%rw_interval == 0)) {
        reweight();
    }

    // Clip weights between weight_min and weight_max
    for (int i = 0; i < weight.size(); i++) {
        weight(i) = weight(i) < weight_min ? weight_min:weight(i);
        weight(i) = weight(i) > weight_max ? weight_max:weight(i);
    }

    // cout << "            " << name << "  --  " << weight.transpose() << "  --  " << gamma.transpose() <<endl;

    U0 = U1;
    Y0 = Y1;

}


void Operator::add2b(VectorXd &b)
{
    Ax_temp.setZero();
    x_temp.setZero();
    
    if (Nc == 1) {
        Ax_temp = U0 + weight(0)*Y0;
    } else {
        Ax_temp.array() = U0.array() + weight.array()*Y0.array();
    }
    transpose(Ax_temp, x_temp, false, 2);
    b += x_temp;
}


void Operator::add2AtAx(VectorXd &X, VectorXd &out)
{
    Ax_temp.setZero();
    x_temp.setZero();
    forward(X, Ax_temp, true, 0, false);
    transpose(Ax_temp, x_temp, false, 2);
    out += x_temp;
}

void Operator::set_inv_vec(VectorXd &inv_vec_in)
{
    inv_vec = inv_vec_in;
}

void Operator::set_fixer(VectorXd &fixer_in)
{
    fixer = fixer_in;
}