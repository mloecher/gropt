#ifndef OP_MOMENTS_H
#define OP_MOMENTS_H

#include <iostream> 
#include <string>
#include <Eigen/Dense>
#include "op_main.h"

using namespace Eigen;
using namespace std; 

class Op_Moments : public Operator
{  
    public:
        int N_moments;
        MatrixXd A;
        MatrixXd moment_params;
        VectorXd mod;

        Op_Moments(int N, double dt, int Nc);
        virtual void set_params_zeros(int N_moments, double moment_tol);
        virtual void set_params(int N_moments, double* moment_params);
        virtual void set_params(MatrixXd &moment_params_in);
        virtual void forward(VectorXd &X, VectorXd &out, bool apply_weight, int norm, bool no_balance);
        virtual void transpose(VectorXd &X, VectorXd &out, bool apply_weight, int norm);
        virtual void prep_A();
        virtual void prox(VectorXd &X);
        virtual void set_inv_vec(VectorXd &inv_vec_in);

};


#endif