#ifndef OP_EDDYSPECT_H
#define OP_EDDYSPECT_H

#include <iostream> 
#include <string>
#include "Eigen/Dense"
#include "op_main.h"

using namespace Eigen;
using namespace std; 

class Op_EddySpect : public Operator
{  
    protected:
        int Neddy;
        double max_lam;
        MatrixXd A;
        VectorXd mod;

        
        enum E_MODE {
            E_MODE_END = 0, // Only compute the final instantaneuous eddy current
            E_MODE_SUM = 1  // Calculate the sum of all eddy currents at all time points
        };
        E_MODE mode;


    public:
        Op_EddySpect(int N, int Naxis, double dt, int Neddy_in);
        // void set_params(int Neddy_in);
        void prep_A();
        // void check(VectorXd &X, int iiter);
        virtual void forward(VectorXd &X, VectorXd &out, bool apply_weight, int norm, bool no_balance);
        virtual void transpose(VectorXd &X, VectorXd &out, bool apply_weight, int norm);
        // virtual void prox(VectorXd &X);
        virtual void get_obj(VectorXd &X, int iiter);
};


#endif