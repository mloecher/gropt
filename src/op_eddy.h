#ifndef OP_EDDY_H
#define OP_EDDY_H

#include <iostream> 
#include <string>
#include "Eigen/Dense"
#include "op_main.h"

using namespace Eigen;
using namespace std; 

class Op_Eddy : public Operator
{  
    public:
        int Neddy;
        int Nlam;
        MatrixXd A;
        VectorXd mod;

        Op_Eddy(int N, int Naxis, double dt, int Nc);
        virtual void forward(VectorXd &X, VectorXd &out, bool apply_weight, int norm, bool no_balance);
        virtual void transpose(VectorXd &X, VectorXd &out, bool apply_weight, int norm);
        virtual void prep_A(double lam, double tol_in);
        virtual void prep_A(VectorXd &lam_in, double tol_in);
        virtual void prep_A(double lam, int eddy_start, int eddy_stop, double tol_in);
        virtual void prep_A(double lam, int eddy_stop, double tol_in);
        virtual void prep_A(VectorXd &lam_in, int eddy_stop, double tol_in);
        virtual void prox(VectorXd &X);
};


#endif