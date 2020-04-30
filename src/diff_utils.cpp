#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <typeinfo>

#include "op_main.h"
#include "op_moments.h"
#include "op_bval.h"
#include "op_slew.h"
#include "op_gradient.h"
#include "gropt_params.h"
#include "diff_utils.h"

void simple_diff(GroptParams &gparams, double dt,
                double T_90, double T_180, double T_readout, double TE,
                double gmax, double smax, int N_moments, double moment_tol, bool siemens_diff)
{ 
    int N = (int)((TE-T_readout)/dt) + 1;
    if (siemens_diff) {N -= 1;} // Our Siemens seq has more specific timings

    int ind_inv = (int)(TE/2.0/dt);
    
    VectorXd inv_vec;
    inv_vec.setOnes(N);
    for(int i = ind_inv; i < N; i++) {
        inv_vec(i) = -1.0;
    }

    int ind_90_end, ind_180_start, ind_180_end;
    VectorXd set_vals;
    set_vals.setOnes(N);
    set_vals.array() *= -9999999.0;

    if (siemens_diff) {  // Code specific for our Siemens sequence, it has tighter tolerances
        ind_90_end = floor(T_90/dt);
        ind_180_start = ind_inv - T_180/dt/2 - 1;
        ind_180_end = ind_inv + T_180/dt/2 - 1;
        for(int i = 0; i < ind_90_end; i++) {
            set_vals(i) = 0.0;
        }
        for(int i = ind_180_start; i <= ind_180_end; i++) {
            set_vals(i) = 0.0;
        }
        set_vals(0) = 0.0;
        set_vals(N-1) = 0.0;
    } else {  // Default case, makes G=0 segments as wide as possible for rounding, safest possible, less efficient
        ind_90_end = ceil(T_90/dt);
        ind_180_start = floor((TE/2.0 - T_180/2.0)/dt);
        ind_180_end = ceil((TE/2.0 + T_180/2.0)/dt);

        for(int i = 0; i <= ind_90_end; i++) {
            set_vals(i) = 0.0;
        }
        for(int i = ind_180_start; i <= ind_180_end; i++) {
            set_vals(i) = 0.0;
        }
        set_vals(0) = 0.0;
        set_vals(N-1) = 0.0;
    }

    VectorXd fixer;
    fixer.setOnes(N);
    for(int i = 0; i < N; i++) {
        if (set_vals(i) > -10000) {
            fixer(i) = 0.0;
        }
    }

    // cout << "N = " << N << endl << endl;
    // cout << "ind_inv = " << ind_inv << "  ind_90_end = " << ind_90_end << "  ind_180_start = " << ind_180_start << "  ind_180_end = " << ind_180_end << endl << endl;

    // Use new so we don't de-allocate later, this means we need to delete later!
    Op_Moments *opM = new Op_Moments(N, dt, N_moments);
    Op_Slew *opS = new Op_Slew(N, dt);
    Op_Gradient *opG = new Op_Gradient(N, dt);
    Op_BVal *opB = new Op_BVal(N, dt);

    opM->set_params_zeros(N_moments, moment_tol);
    opS->set_params(smax);
    opG->set_params(gmax, set_vals);
    opB->weight(0) = -1.0;

    vector<Operator*> all_op;
    all_op.push_back(opG);
    all_op.push_back(opS);
    all_op.push_back(opM);

    vector<Operator*> all_obj;
    all_obj.push_back(opB);

    VectorXd X;
    X.setOnes(N);
    X.array() *= inv_vec.array() * fixer.array() * gmax/10.0;

    gparams.N = N;
    gparams.X0 = X;
    gparams.all_op = all_op; 
    gparams.all_obj = all_obj;
    gparams.inv_vec = inv_vec;
    gparams.fixer = fixer;
    gparams.set_vals = set_vals;

    gparams.set_vecs();
    gparams.defaults_diffusion();
}