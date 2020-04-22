#ifndef OP_MAIN_H
#define OP_MAIN_H

#include <iostream> 
#include <string>
#include <Eigen/Dense>

using namespace Eigen;
using namespace std; 

class Operator 
{  
    public:   
        int N;
        double dt;
        string name;

        VectorXd weight;
        VectorXd gamma;
        
        double cushion;
        VectorXd tol0;
        VectorXd tol;
        VectorXd target;

        bool row_constraints; // Are there multiple individual values constraints?
        int Nc; // Number of constraints within the operator
        int Ax_size; // Vector size of Ax
        VectorXd spec_norm2;

        bool balanced;
        VectorXd balance_mod;

        VectorXd x_temp;
        VectorXd Ax_temp;

        VectorXd Y0;
        VectorXd Y1;
        VectorXd U0;
        VectorXd U1;

        VectorXd s;
        VectorXd xbar;

        VectorXd Uhat00;
        VectorXd U00;
        VectorXd s00;
        VectorXd Y00;

        VectorXd r_feas;
        VectorXd feas_temp;
        VectorXd feas_check;
        
        double e_corr;
        bool do_rw;
        double rw_eps;
        double rw_scalelim;
        int rw_interval;

        double weight_min;
        double weight_max;

        VectorXd fixer;
        VectorXd inv_vec;

        // Vectors to pre-allocate for the reweighting
        VectorXd uhat1;
        VectorXd duhat;
        VectorXd du;
        VectorXd dhhat;
        VectorXd dghat;

        MatrixXd hist_check;
        MatrixXd hist_feas;
        MatrixXd hist_obj;

        double current_obj;
        
        Operator(int N, double dt, int Nc, int Ax_size, bool row_constraints);
        void allocate_rwvecs();    
        void reweight();
        void update(VectorXd &X, int iiter);
        void add2b(VectorXd &b);
        void add2AtAx(VectorXd &X, VectorXd &out);
        virtual void init(VectorXd &X); 
        virtual void prep_y(VectorXd &X); 
        virtual void forward(VectorXd &X, VectorXd &out, bool apply_weight, int norm, bool no_balance);
        virtual void transpose(VectorXd &X, VectorXd &out, bool apply_weight, int norm);
        virtual void prox(VectorXd &X);
        virtual void check(VectorXd &X, int iiter);
        virtual void get_obj(VectorXd &X, int iiter);
        virtual void set_inv_vec(VectorXd &inv_vec_in);
        virtual void set_fixer(VectorXd &fixer_in);
        virtual void change_cushion(double cushion_in);
        
};

#endif