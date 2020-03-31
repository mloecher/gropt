#ifndef OP_GRADIENT_H
#define OP_GRADIENT_H

#include <iostream> 
#include <string>
#include <Eigen/Dense>
#include "op_main.h"

using namespace Eigen;
using namespace std; 

class Op_Gradient : public Operator
{  
    protected:
        double gmax;
        VectorXd set_vals;

    public:
        Op_Gradient(int N, double dt);
        virtual void set_params(double gmax_in);
        virtual void set_params(double gmax_in, VectorXd &set_vals_in);
        virtual void forward(VectorXd &X, VectorXd &out, bool apply_weight, int norm, bool no_balance);
        virtual void transpose(VectorXd &X, VectorXd &out, bool apply_weight, int norm);
        virtual void prox(VectorXd &X);

};


#endif