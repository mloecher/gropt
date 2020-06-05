#ifndef OP_BVAL_H
#define OP_BVAL_H

#include <iostream> 
#include <string>
#include <Eigen/Dense>
#include "op_main.h"

using namespace Eigen;
using namespace std; 

class Op_BVal : public Operator
{  
    protected:
        double GAMMA;
        double MAT_SCALE;
        double bval0;

    public:
        Op_BVal(int N, int Naxis, double dt);
        void set_params(double bval_in);
        void check(VectorXd &X, int iiter);
        virtual void forward(VectorXd &X, VectorXd &out, bool apply_weight, int norm, bool no_balance);
        virtual void transpose(VectorXd &X, VectorXd &out, bool apply_weight, int norm);
        virtual void prox(VectorXd &X);
        virtual void get_obj(VectorXd &X, int iiter);

};


#endif