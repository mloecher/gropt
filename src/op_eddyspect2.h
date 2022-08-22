#ifndef OP_EDDYSPECT2_H
#define OP_EDDYSPECT2_H

#include <iostream> 
#include <string>
#include "Eigen/Dense"
#include "op_main.h"

using namespace Eigen;
using namespace std; 

class Op_EddySpect2 : public Operator
{  
    protected:
        int Neddy;
        double max_lam;
        int i_start;
        int i_stop;
        MatrixXd A;
        VectorXd mod;
        VectorXd target_spect;

        
        enum E_MODE {
            E_MODE_END = 0, // Only compute the final instantaneuous eddy current
            E_MODE_SUM = 1  // Calculate the sum of all eddy currents at all time points
        };
        E_MODE mode;


    public:
        Op_EddySpect2(int N, int Naxis, double dt, int Neddy_in, int i_start, int i_stop);
        // void set_params(int Neddy_in);
        void prep_A();
        // void check(VectorXd &X, int iiter);
        virtual void forward(VectorXd &X, VectorXd &out, bool apply_weight, int norm, bool no_balance);
        virtual void transpose(VectorXd &X, VectorXd &out, bool apply_weight, int norm);
        // virtual void prox(VectorXd &X);
        virtual void get_obj(VectorXd &X, int iiter);
        void set_spect(double *spect_in);
        virtual void obj_add2b(VectorXd &b);
};


#endif