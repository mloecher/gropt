#ifndef GROPT_PARAMS_H
#define GROPT_PARAMS_H

#include <iostream> 
#include <string>
#include <vector>
#include <Eigen/Dense>
#include "op_main.h"

using namespace Eigen;
using namespace std; 

class GroptParams
{
    public:
        vector<Operator*> all_op;
        vector<Operator*> all_obj;

        VectorXd inv_vec;
        VectorXd set_vals;
        VectorXd fixer;

        VectorXd X0;

        double dt;
        double N;

        double gmax;
        double smax;

        int grw_interval;
        int grw_start;
        double grw_scale;

        double rw_scalelim;
        double rw_interval;
        double rw_eps;
        double e_corr;
        double weight_min;
        double weight_max;

        double cushion;

        int N_iter;  // Maximum number of outer iterations
        int N_feval; // Maximum number of function evaluations (includes CG iterations)

        int cg_niter;
        double cg_resid_tol;
        double cg_abs_tol;

        double d_obj_thresh;

        double total_n_feval;
        int last_iiter;

        GroptParams();
        void update_vals();
        void defaults_diffusion();
};

#endif