#ifndef OP_BTENSOR_H
#define OP_BTENSOR_H

#include <iostream> 
#include <string>
#include "Eigen/Dense"
#include "op_main.h"

using namespace Eigen;
using namespace std; 

class Op_BTensor : public Operator
{  
    protected:
        double GAMMA;
        double MAT_SCALE;
        double bval0;

    public:
        Op_BTensor(int N, int Naxis, double dt);
        void set_params(double bval_in);
        void set_params(double bval_in0, double bval_in1, double bval_in2);
        void check(VectorXd &X, int iiter);
        void test(VectorXd &X);
        virtual void forward(VectorXd &X, VectorXd &out, bool apply_weight, int norm, bool no_balance);
        virtual void transpose(VectorXd &X, VectorXd &out, bool apply_weight, int norm);
        virtual void prox(VectorXd &X);
        virtual void get_obj(VectorXd &X, int iiter);
        virtual void get_feas(VectorXd &s, int iiter);

};


#endif