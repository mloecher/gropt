#ifndef OP_MAXWELL_H
#define OP_MAXWELL_H

#include <iostream> 
#include <string>
#include "Eigen/Dense"
#include "op_main.h"

using namespace Eigen;
using namespace std; 

class Op_Maxwell : public Operator
{  
    protected:
        double tolerance;
        // VectorXd set_vals;

        int ind0;
        int ind1;
        int ind2;

    public:
        Op_Maxwell(int N, int Naxis, double dt);
        virtual void set_params(double tol_in, int ind0_in, int ind1_in, int ind2_in);
        virtual void set_params(double tol_in, VectorXd &set_vals_in, int ind0_in, int ind1_in, int ind2_in);
        virtual void forward(VectorXd &X, VectorXd &out, bool apply_weight, int norm, bool no_balance);
        virtual void transpose(VectorXd &X, VectorXd &out, bool apply_weight, int norm);
        virtual void prox(VectorXd &X);
        virtual void check(VectorXd &X, int iiter);

};


#endif