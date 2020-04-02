#ifndef CG_ITER_H
#define CG_ITER_H

#include <iostream> 
#include <string>
#include <vector>
#include <Eigen/Dense>
#include "op_main.h"

using namespace Eigen;
using namespace std; 

void get_Ax(vector<Operator*> all_op, vector<Operator*> all_obj, VectorXd &X, VectorXd &Ax);

class CG_Iter
{
    public:
        VectorXd b;
        
        VectorXd Ax;
        VectorXd Ap;
        VectorXd x1;
        VectorXd r;
        VectorXd p;

        int N;
        int max_iter;
        double resid_tol;
        double abs_tol;
        int n_iter;

        CG_Iter(int N, int max_iter, double resid_tol, double abs_tol);
        VectorXd do_CG(vector<Operator*> all_op, vector<Operator*> all_obj, VectorXd &x0);
        int get_n_iter();
};

#endif