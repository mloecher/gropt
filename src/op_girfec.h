#ifndef OP_GIRFEC_H
#define OP_GIRFEC_H

#include <iostream> 
#include <string>
#include "Eigen/Dense"
#include "op_main.h"
#include "pocketfft_hdronly.h"

using namespace Eigen;
using namespace std; 

class Op_GirfEC : public Operator
{  
    public:
        unsigned long long N_FT;
        VectorXcd ft_c_vec0;
        VectorXcd ft_c_vec1;
        VectorXcd H;

        VectorXd girf_win;

        pocketfft::shape_t shape;
        pocketfft::stride_t stride_d;
        pocketfft::stride_t stride_cd;
        pocketfft::shape_t axes;


        Op_GirfEC(int N, int Naxis, double dt, VectorXcd &H_in, VectorXd &girf_win_in, int N_H) ;
        virtual void forward(VectorXd &X, VectorXd &out, bool apply_weight, int norm, bool no_balance);
        virtual void transpose(VectorXd &X, VectorXd &out, bool apply_weight, int norm);
        virtual void prox(VectorXd &X);
        virtual void get_obj(VectorXd &X, int iiter);

};


#endif