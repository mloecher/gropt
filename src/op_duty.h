#ifndef OP_DUTY_H
#define OP_DUTY_H

#include <iostream> 
#include <string>
#include <Eigen/Dense>
#include "op_main.h"

using namespace Eigen;
using namespace std; 

class Op_Duty : public Operator
{  
    public:
        Op_Duty(int N, int Naxis, double dt);
        virtual void forward(VectorXd &X, VectorXd &out, bool apply_weight, int norm, bool no_balance);
        virtual void transpose(VectorXd &X, VectorXd &out, bool apply_weight, int norm);
        virtual void prox(VectorXd &X);
        virtual void get_obj(VectorXd &X, int iiter);

};


#endif