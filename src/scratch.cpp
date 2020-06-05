#include <iostream>
#include <fstream>
#include <vector>
#include <Eigen/Dense>
#include <typeinfo>
#ifdef USE_CHRONO
    #include <chrono> 
#endif

#include "op_main.h"
#include "op_moments.h"
#include "op_bval.h"
#include "op_slew.h"
#include "op_gradient.h"
#include "op_duty.h"
#include "cg_iter.h"
#include "gropt_params.h"
#include "optimize.h"
#include "diff_utils.h"

using namespace Eigen;
using namespace std;

void write_binary_vectorxd(const char* filename, const VectorXd& data){
    ofstream out(filename, ios::out | ios::binary | ios::trunc);
    double size = data.size();
    out.write((char*) (&size), sizeof(double));
    out.write((char*) data.data(), size*sizeof(double));
    out.close();
}

void oned_flowcomp() 
{
    double dt = 20e-6;
    double T = 3.0e-3;
    double gmax = .04;
    double smax = 100.0;

    int N = (int)(T/dt) + 1;

    VectorXd inv_vec;
    inv_vec.setOnes(N);

    VectorXd set_vals;
    set_vals.setOnes(N);
    set_vals.array() *= -9999999.0;
    set_vals(0) = 0.0;
    set_vals(N-1) = 0.0;

    VectorXd fixer;
    fixer.setOnes(N);
    for(int i = 0; i < N; i++) {
        if (set_vals(i) > -10000) {
            fixer(i) = 0.0;
        }
    }

    int Naxis = 1;
    
    int N_moments = 2;
    Op_Moments *opM = new Op_Moments(N, Naxis, dt, N_moments);
    MatrixXd moments(N_moments,7);
    moments << 0, 0, 0, 0, 0, 11.74, 1e-6,
               0, 1, 0, 0, 0, 0.0, 1e-6;
    opM->set_params(moments);

    // int N_moments = 1;
    // Op_Moments *opM = new Op_Moments(N, dt, N_moments);
    // MatrixXd moments(N_moments,7);
    // moments << 0, 0, 0, 0, 0, 11.74, 1e-6;
    // opM->set_params(moments);

    Op_Slew *opS = new Op_Slew(N, Naxis, dt);
    Op_Gradient *opG = new Op_Gradient(N, Naxis, dt);
    Op_Duty *opD = new Op_Duty(N, Naxis, dt);

    
    opS->set_params(smax);
    opG->set_params(gmax, set_vals);
    opD->weight(0) = 1.0;
    
    vector<Operator*> all_op;
    all_op.push_back(opG);
    all_op.push_back(opS);
    all_op.push_back(opM);

    vector<Operator*> all_obj;
    all_obj.push_back(opD);
    

    VectorXd X;
    X.setOnes(N);
    X.array() *= fixer.array() * gmax/10.0;

    GroptParams gparams;
    gparams.N = N;
    gparams.X0 = X;
    gparams.all_op = all_op; 
    gparams.all_obj = all_obj;
    gparams.inv_vec = inv_vec;
    gparams.fixer = fixer;
    gparams.set_vals = set_vals;
    gparams.verbose = 3;

    gparams.d_obj_thresh = 1e-5;

    gparams.set_vecs();
    gparams.update_vals();

    VectorXd out;
    optimize(gparams, out);

    cout << "  !!!!!!!!!  " << " -- Done -- " << "  !!!!!!!!!  " << endl << endl;

    for (int i = 0; i < gparams.all_op.size(); i++) {
        VectorXd temp;
        temp.setZero(all_op[i]->Y0.size());
        gparams.all_op[i]->forward(out, temp, false, 0, true);

        cout << "***** " << gparams.all_op[i]->name << " *****" << endl
        << "  --  Checks: " << gparams.all_op[i]->hist_check.col(gparams.last_iiter).transpose() << endl
        // << "  --  Vals: " << temp.transpose() << endl
        << endl;
    }
    
    write_binary_vectorxd("../python/testing/temp.bin", out);
    // cout << "Done!" << endl << out.transpose() << endl;

}


void threed_flowcomp() 
{
    double dt = 20e-6;
    double T = 2.555e-3;
    double gmax = .03;
    double smax = 100.0;

    int N = (int)(T/dt) + 1;

    int Naxis = 3;

    VectorXd inv_vec;
    inv_vec.setOnes(Naxis*N);

    VectorXd set_vals;
    set_vals.setOnes(Naxis*N);
    set_vals.array() *= -9999999.0;
    for (int j = 0; j < Naxis; j++) {
        set_vals((j*N)) = 0.0;
        set_vals((j*N)+(N-1)) = 0.0;
    }

    VectorXd fixer;
    fixer.setOnes(Naxis*N);
    for(int i = 0; i < Naxis*N; i++) {
        if (set_vals(i) > -10000) {
            fixer(i) = 0.0;
        }
    }
    
    // int N_moments = 2;
    // Op_Moments *opM = new Op_Moments(N, Naxis, dt, N_moments);
    // MatrixXd moments(N_moments,7);
    // moments << 0, 0, 0, 0, 0, 11.74, 1e-6,
    //            0, 1, 0, 0, 0, 0.0, 1e-6;
    // opM->set_params(moments);

    // int N_moments = 1;
    // Op_Moments *opM = new Op_Moments(N, dt, N_moments);
    // MatrixXd moments(N_moments,7);
    // moments << 0, 0, 0, 0, 0, 11.74, 1e-6;
    // opM->set_params(moments);

    int N_moments = 7;
    Op_Moments *opM = new Op_Moments(N, Naxis, dt, N_moments);
    MatrixXd moments(N_moments,7);
    moments << 0, 0, 0, 0, 0, 11.74, 1e-6,
               0, 1, 0, 0, 0, 0.0, 1e-6,
               1, 0, 0, 0, 0, -11.74, 1e-6,
               1, 1, 0, 0, 0, 0.0, 1e-6,
               2, 0, 0, 0, 0, 11.74, 1e-6,
               2, 1, 0, 0, 0, 0.0, 1e-6,
               2, 2, 0, 0, 0, 0.0, 1e-6;
    opM->set_params(moments);

    Op_Slew *opS = new Op_Slew(N, Naxis, dt);
    Op_Gradient *opG = new Op_Gradient(N, Naxis, dt);
    Op_Duty *opD = new Op_Duty(N, Naxis, dt);
    
    opS->set_params(smax);
    opG->set_params(gmax, set_vals);
    opD->weight(0) = 0.0;
    
    vector<Operator*> all_op;
    all_op.push_back(opG);
    all_op.push_back(opS);
    all_op.push_back(opM);

    vector<Operator*> all_obj;
    all_obj.push_back(opD);

    VectorXd X;
    X.setOnes(Naxis*N);
    X.array() *= fixer.array() * gmax/10.0;

    GroptParams gparams;
    gparams.N = N;
    gparams.Naxis = Naxis;
    gparams.X0 = X;
    gparams.all_op = all_op; 
    gparams.all_obj = all_obj;
    gparams.inv_vec = inv_vec;
    gparams.fixer = fixer;
    gparams.set_vals = set_vals;
    gparams.verbose = 3;
    
    gparams.N_iter = 5000;
    gparams.N_feval = 50000;
    gparams.d_obj_thresh = 1e-4;

    gparams.set_vecs();
    gparams.update_vals();

    VectorXd out;
    optimize(gparams, out);

    cout << "  !!!!!!!!!  " << " -- Done -- " << "  !!!!!!!!!  " << endl << endl;

    for (int i = 0; i < gparams.all_op.size(); i++) {
        VectorXd temp;
        temp.setZero(all_op[i]->Y0.size());
        gparams.all_op[i]->forward(out, temp, false, 0, true);

        cout << "***** " << gparams.all_op[i]->name << " *****" << endl
        << "  --  Checks: " << gparams.all_op[i]->hist_check.col(gparams.last_iiter).transpose() << endl
        // << "  --  Vals: " << temp.transpose() << endl
        << endl;
    }
    
    write_binary_vectorxd("../python/testing/temp.bin", out);
    // cout << "Done!" << endl << out.transpose() << endl;

}

