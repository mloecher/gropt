#ifndef OP_SLEW_H
#define OP_SLEW_H

#include <iostream> 
#include <string>
#include <Eigen/Dense>
#include "op_main.h"

using namespace Eigen;
using namespace std; 

class Op_Slew : public Operator
{  
    protected:
        double smax;

    public:
        Op_Slew(int N, double dt);
        virtual void set_params(double smax_in);
        virtual void forward(VectorXd &X, VectorXd &out, bool apply_weight, int norm, bool no_balance);
        virtual void transpose(VectorXd &X, VectorXd &out, bool apply_weight, int norm);
        virtual void prox(VectorXd &X);

};


#endif