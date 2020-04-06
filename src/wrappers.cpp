#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <typeinfo>
#include <chrono> 

#include "op_main.h"
#include "op_moments.h"
#include "op_bval.h"
#include "op_slew.h"
#include "op_gradient.h"
#include "cg_iter.h"
#include "gropt_params.h"
#include "optimize.h"

using namespace Eigen;
using namespace std;


void python_wrapper_v1(double *params0, double *params1, double **out0, double **out1, double **out2, int **outsize) {
    cout << "In python_wrapper_v1:" << endl << endl;
    
    double dt = params0[0];
    double T_90 = params0[1];
    double T_180 = params0[2];
    double T_readout = params0[3];
    double TE = params0[4];

    int N = (int)((TE-T_readout)/dt) + 1;

    int ind_inv = (int)(TE/2.0/dt);
    VectorXd inv_vec;
    inv_vec.setOnes(N);
    for(int i = ind_inv; i < N; i++) {
        inv_vec(i) = -1.0;
    }

    int ind_90_end = ceil(T_90/dt);
    int ind_180_start = floor((TE/2.0 - T_180/2.0)/dt);
    int ind_180_end = ceil((TE/2.0 + T_180/2.0)/dt);

    VectorXd set_vals;
    set_vals.setOnes(N);
    set_vals.array() *= -9999999.0;
    for(int i = 0; i <= ind_90_end; i++) {
        set_vals(i) = 0.0;
    }
    for(int i = ind_180_start; i <= ind_180_end; i++) {
        set_vals(i) = 0.0;
    }
    set_vals(0) = 0.0;
    set_vals(N-1) = 0.0;

    VectorXd fixer;
    fixer.setOnes(N);
    for(int i = 0; i < N; i++) {
        if (set_vals(i) > -10000) {
            fixer(i) = 0.0;
        }
    }

    cout << "N = " << N << endl << endl;
    cout << "ind_inv = " << ind_inv << "  ind_90_end = " << ind_90_end << "  ind_180_start = " << ind_180_start << "  ind_180_end = " << ind_180_end << endl << endl;

    int N_moments = params0[5];
    double moment_tol = params0[6];

    Op_Moments opM(N, dt, 3);
    Op_Slew opS(N, dt);
    Op_Gradient opG(N, dt);
    Op_BVal opB(N, dt);

    opM.set_inv_vec(inv_vec);
    opM.set_fixer(fixer);
    opS.set_inv_vec(inv_vec);
    opS.set_fixer(fixer);
    opG.set_inv_vec(inv_vec);
    opG.set_fixer(fixer);
    opB.set_inv_vec(inv_vec);
    opB.set_fixer(fixer);

    MatrixXd moments;
    if (N_moments == 1) {
        moments.setZero(1,7);
        moments << 0, 0, 0, 0, 0, 0.0, moment_tol;
    } else if (N_moments == 2) {
        moments.setZero(2,7);
        moments << 0, 0, 0, 0, 0, 0.0, moment_tol,
                   0, 1, 0, 0, 0, 0.0, moment_tol;
    } else if (N_moments == 3) {
        moments.setZero(3,7);
        moments << 0, 0, 0, 0, 0, 0.0, moment_tol,
                0, 1, 0, 0, 0, 0.0, moment_tol,
                0, 2, 0, 0, 0, 0.0, moment_tol;
    } else {
        moments.setZero(1,7);
        moments << 0, 0, 0, 0, 0, 0.0, moment_tol;
    }

    double gmax = params0[7];
    double smax = params0[8];

    opM.set_params(moments);
    opS.set_params(smax);
    opG.set_params(gmax, set_vals);
    opB.weight(0) = -1.0;

    vector<Operator*> all_op;
    all_op.push_back(&opG);
    all_op.push_back(&opS);
    all_op.push_back(&opM);

    vector<Operator*> all_obj;
    all_obj.push_back(&opB);

    VectorXd X;
    X.setOnes(N);
    X.array() *= inv_vec.array() * fixer.array() * gmax/10.0;

    GroptParams gparams;
    gparams.N = N;
    gparams.X0 = X;
    gparams.all_op = all_op; 
    gparams.all_obj = all_obj;
    gparams.inv_vec = inv_vec;
    gparams.fixer = fixer;
    gparams.set_vals = set_vals;

    gparams.cushion = params0[9];
    gparams.rw_scalelim = params0[10];
    gparams.rw_interval = params0[11];
    gparams.rw_eps = params0[12];
    gparams.e_corr = params0[13];
    gparams.weight_min = params0[14];
    gparams.weight_max = params0[15];

    gparams.update_vals();

    VectorXd out;
    optimize(gparams, out);

    int N_out0 = out.size();
    *out0 = new double[N_out0];
    for(int i = 0; i < N_out0; i++) {
        out0[0][i] = out(i);
    }

    *outsize = new int[3];
    outsize[0][0] = N_out0;

    cout << "Done python_wrapper_v1!" << endl << endl;

}


int main()
{
    return 0;
}