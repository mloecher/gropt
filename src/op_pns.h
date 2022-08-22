#ifndef OP_PNS_H
#define OP_PNS_H

#include <iostream> 
#include <string>
#include "Eigen/Dense"
#include "op_main.h"

using namespace Eigen;
using namespace std; 

class Op_PNS : public Operator
{  
    protected:
        double stim_thresh;
        VectorXd coeff;

    public:
        Op_PNS(int N, int Naxis, double dt);
        virtual void set_params(double stim_thresh_in);
        virtual void forward(VectorXd &X, VectorXd &out, bool apply_weight, int norm, bool no_balance);
        virtual void transpose(VectorXd &X, VectorXd &out, bool apply_weight, int norm);
        virtual void prox(VectorXd &X);
        virtual void check(VectorXd &X, int iiter);

};


#endif