#include <iostream>
#include <fstream>
#include <vector>
#include "Eigen/Dense"
#include <typeinfo>
#ifdef USE_CHRONO
    #include <chrono> 
#endif

#include "op_main.h"
#include "op_maxwell.h"
#include "op_moments.h"
#include "op_eddy.h"
#include "op_eddyspect.h"
#include "op_eddyspect2.h"
#include "op_bval.h"
#include "op_btensor.h"
#include "op_slew.h"
#include "op_gradient.h"
#include "op_pns.h"
#include "op_duty.h"
#include "op_acoustic.h"
#include "op_girfec.h"
#include "cg_iter.h"
#include "gropt_params.h"
#include "optimize.h"
#include "diff_utils.h"

#include "pocketfft_hdronly.h"

using namespace Eigen;
using namespace std;

// This is a call for general purpose testing, i.e. for testing the optimization hyperparameters
void python_wrapper_v1(double *params0, double *params1, double **out0, double **out1, double **out2, int **outsize) 
{
    cout << "In python_wrapper_v1:" << endl << endl;
    
    double dt = params0[0];
    double T_90 = params0[1];
    double T_180 = params0[2];
    double T_readout = params0[3];
    double TE = params0[4];

    int N_moments = params0[5];
    double moment_tol = params0[6];

    double gmax = params0[7];
    double smax = params0[8];

    #ifdef USE_CHRONO
        auto t_start0 = chrono::steady_clock::now();
    #endif 


    GroptParams gparams;
    simple_diff(gparams, dt, T_90, T_180, T_readout, TE, gmax, smax, N_moments, moment_tol, false);

    gparams.cushion = params0[9];
    gparams.rw_scalelim = params0[10];
    gparams.rw_interval = params0[11];
    gparams.rw_eps = params0[12];
    gparams.e_corr = params0[13];
    gparams.weight_min = params0[14];
    gparams.weight_max = params0[15];
    gparams.d_obj_thresh = params0[16];

    gparams.grw_interval = params0[17];
    gparams.grw_start = params0[18];
    gparams.grw_scale = params0[19];

    gparams.cg_niter = params0[20];
    gparams.cg_resid_tol = params0[21];
    gparams.cg_abs_tol = params0[22];

    gparams.update_vals();

    int n_timeruns = 10;
    
    #ifdef USE_CHRONO
        auto t_start1 = chrono::steady_clock::now();
    #endif
    
    VectorXd out;
    for (int i_rep = 0; i_rep < n_timeruns; i_rep++) { 
        optimize(gparams, out);
    }

    #ifdef USE_CHRONO
        auto t_end = chrono::steady_clock::now(); 
        double time_taken0 = chrono::duration_cast<chrono::microseconds>(t_end - t_start0).count(); 
        double time_taken1 = chrono::duration_cast<chrono::microseconds>(t_end - t_start1).count() / (double)n_timeruns;
    #else
        double time_taken0 = 0.0;
        double time_taken1 = 0.0;
    #endif

    cout << "Time taken 0 by program is : " << time_taken0 << " microsec" << endl; 
    cout << "Time taken 1 by program is : " << time_taken1 << " microsec" << endl; 

    int N_out0 = out.size();
    *out0 = new double[N_out0];
    for(int i = 0; i < N_out0; i++) {
        out0[0][i] = out(i);
    }

    int N_out1 = 16;
    *out1 = new double[N_out1];
    out1[0][0] = gparams.total_n_feval;
    out1[0][1] = gparams.all_obj[0]->current_obj;
    out1[0][2] = time_taken1 * 1e-6;
    out1[0][3] = gparams.last_iiter;

    int N_out2 = 160;
    VectorXd v_out2;
    v_out2.setOnes(N_out2);

    interp_vec2vec(out, v_out2);

    *out2 = new double[N_out2];
    for(int i = 0; i < N_out2; i++) {
        out2[0][i] = v_out2(i);
    }


    *outsize = new int[3];
    outsize[0][0] = N_out0;
    outsize[0][1] = N_out1;
    outsize[0][2] = N_out2;
    

    cout << "Done python_wrapper_v1!" << endl << endl;
}


// This is the call for our Siemens diffusion sequence
void gropt_diff_seq(double **G_out, int *N_out, int verbose,
                    double dt0, double dt_out, double gmax, double smax, double TE,
                    double T_readout, double T_90, double T_180, int MMT)
{
    
    int N_moments = MMT;
    double moment_tol = 1.0e-3;
    bool siemens_diff = false;

    GroptParams gparams;
    simple_diff(gparams, dt0, T_90, T_180, T_readout, TE, gmax, smax, N_moments, moment_tol, siemens_diff);
    
    #ifdef USE_CHRONO
        auto t_start1 = chrono::steady_clock::now();
    #endif

    VectorXd out;
    optimize(gparams, out);

    #ifdef USE_CHRONO
        auto t_end1 = chrono::steady_clock::now(); 
        double time_taken1 = chrono::duration_cast<chrono::microseconds>(t_end1 - t_start1).count(); 
    #else
        double time_taken1 = 0.0;
    #endif

    int N_interp = (int)((TE-T_readout)/dt_out) + 1;
    if (siemens_diff) {N_interp -= 1;}

    VectorXd out_interp;
    out_interp.setOnes(N_interp);

    interp_vec2vec(out, out_interp);

    *N_out = N_interp;
    *G_out = (double *) malloc(N_interp*sizeof(double));
    for (int i = 0; i < N_interp; i++) {
        G_out[0][i] = out_interp[i];
    }  

    return;
}

// This exactly matches the function call of the old gropt function "run_kernel_diff_fixeddt"
void gropt_legacy(double **G_out, int *N_out, double **ddebug, int verbose, 
                  double dt0, double gmax, double smax, double TE, 
                  int N_moments, double *moments_params, double PNS_thresh, 
                  double T_readout, double T_90, double T_180, int diffmode, double dt_out,
                  int N_eddy, double *eddy_params, double search_bval, double slew_reg, int Naxis)
{
    double dt = dt0;
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

    // This is where I may need to start the loops?  Or I need some type of reinitialization of everything

    Op_Moments *opM = new Op_Moments(N, Naxis, dt, N_moments);
    Op_Slew *opS = new Op_Slew(N, Naxis, dt);
    Op_Gradient *opG = new Op_Gradient(N, Naxis, dt);
    Op_BVal *opB = new Op_BVal(N, Naxis, dt);

    opM->set_inv_vec(inv_vec);
    opM->set_fixer(fixer);
    opS->set_inv_vec(inv_vec);
    opS->set_fixer(fixer);
    opG->set_inv_vec(inv_vec);
    opG->set_fixer(fixer);
    opB->set_inv_vec(inv_vec);
    opB->set_fixer(fixer);

    MatrixXd moments;
    moments.setZero(N_moments,7);
    for (int i = 0; i < N_moments; i++) {
        for (int j = 0; j < 7; j++) {
            moments(i,j) = moments_params[j + i*7];
        }
    }


    opM->set_params(moments);
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

    GroptParams gparams;
    gparams.N = N;
    gparams.X0 = X;
    gparams.all_op = all_op; 
    gparams.all_obj = all_obj;
    gparams.inv_vec = inv_vec;
    gparams.fixer = fixer;
    gparams.set_vals = set_vals;
    
    gparams.update_vals();

    VectorXd out;
    optimize(gparams, out);

    int N_out0 = out.size();
    *G_out = new double[N_out0];
    for(int i = 0; i < N_out0; i++) {
        G_out[0][i] = out(i);
    }

    *N_out = N_out0;

    int N_ddebug = 100;
    *ddebug = new double[N_ddebug];
    for(int i = 0; i < N_ddebug; i++) {
        ddebug[0][i] = 0.0;
    }

    cout << "Done gropt_legacy!" << endl << endl;

}


// This exactly matches the function call of the old gropt function "run_kernel_diff_fixeddt"
void en_code(double dt, double T_90, double T_180, double T_readout, double TE, 
                     int N_moments, double gmax, double smax, double lambda,
                     double **out0, double **out1, double **out2, int **outsize)
{
    double moment_tol = 1.0e-4;
    
    GroptParams gparams;
    gparams.verbose = 2;
    gparams.N_iter = 10000;
    gparams.cg_resid_tol = 1.0e-4;
    
    int N = (int)((TE-T_readout)/dt) + 1;

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
    int Naxis = 1;
    
    Op_Moments *opM = new Op_Moments(N, Naxis, dt, N_moments);
    Op_Slew *opS = new Op_Slew(N, Naxis, dt);
    Op_Gradient *opG = new Op_Gradient(N, Naxis, dt);
    Op_Eddy *opE = new Op_Eddy(N, Naxis, dt, 1);
    opM->set_params_zeros(N_moments, moment_tol);
    opS->set_params(smax);
    opG->set_params(gmax, set_vals);
    opE->prep_A(lambda, 1.0e-4);
    
    

    vector<Operator*> all_op;
    all_op.push_back(opG);
    all_op.push_back(opS);
    all_op.push_back(opM);
    all_op.push_back(opE);


    vector<Operator*> all_obj;
    Op_BVal *opB2 = new Op_BVal(N, Naxis, dt);
    opB2->weight(0) = -1.0;
    all_obj.push_back(opB2);

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

    VectorXd out;
    optimize(gparams, out);

    cout << "Final " << gparams.total_n_feval << endl;

    int N_out0 = out.size();
    *out0 = new double[N_out0];
    for(int i = 0; i < N_out0; i++) {
        out0[0][i] = out(i);
    }

    int N_out1 = 16;
    *out1 = new double[N_out1];
    out1[0][0] = gparams.final_good;
    out1[0][1] = gparams.last_iiter;
    out1[0][2] = gparams.total_n_feval;

    *outsize = new int[3];
    outsize[0][0] = N_out0;
    outsize[0][1] = N_out1;
    // outsize[0][2] = N_out2;

}

// This is just a testing function used to find the optimal way to warmstart iterations
// ToDo: Compare the actual initialization for both perfect warm start and interpolated warm start, see what the exact differences are
void python_wrapper_warmstart_v1(double *params0, double *params1, double **out0, double **out1, double **out2, int **outsize) 
{
    cout << "In python_wrapper_warmstart_v1:" << endl << endl;

    double dt = 400.0e-6;
    double T_90 = 4.0e-3;
    double T_180 = 6.0e-3;
    double T_readout = 12.0e-3;
    double TE = 64.0e-3;

    int N_moments = 3;
    double moment_tol = 1.0e-4;

    double gmax = .04;
    double smax = 50.0;

    
    GroptParams gparams;
    
    simple_diff(gparams, dt, T_90, T_180, T_readout, TE, gmax, smax, N_moments, moment_tol, false);

    #ifdef USE_CHRONO
        auto t_start = chrono::steady_clock::now();
    #endif

    VectorXd out;
    optimize(gparams, out);

    cout << "Final " << gparams.total_n_feval << "  " << gparams.all_obj[0]->current_obj << endl;

    #ifdef USE_CHRONO
        auto t_end = chrono::steady_clock::now(); 
        double time_taken = chrono::duration_cast<chrono::microseconds>(t_end - t_start).count();
    #else
        double time_taken = 0.0;
    #endif 

    cout << "Time taken by optimize1: " << time_taken/1000.0 << " msec" << endl; 

    cout << endl << "***" << endl;
    for (int i = 0; i < gparams.all_op.size(); i++) {
        cout << gparams.all_op[i]->name << "  --  " << gparams.all_op[i]->U0.squaredNorm() << "  --  " << 
        gparams.all_op[i]->Y0.squaredNorm() << "  --  " << gparams.all_op[i]->weight.transpose() << endl;
    }
    cout << "***" << endl << endl;
    
    cout << endl << "-----------------------------------------------------";
    cout << endl << "------------ dt 50 Run -------------" << endl;
    cout << "-----------------------------------------------------" << endl << endl;

    GroptParams gparams_2;
    double dt2 = 200.0e-6;
    simple_diff(gparams_2, dt2, T_90, T_180, T_readout, TE, gmax, smax, N_moments, moment_tol, false);

    #ifdef USE_CHRONO
        t_start = chrono::steady_clock::now();
    #endif

    VectorXd out_2;
    optimize(gparams_2, out_2);

    cout << "Final2 " << gparams_2.total_n_feval << "  " << gparams_2.all_obj[0]->current_obj << endl;

    #ifdef USE_CHRONO
        t_end = chrono::steady_clock::now(); 
        time_taken = chrono::duration_cast<chrono::microseconds>(t_end - t_start).count();
    #else
        time_taken = 0.0;
    #endif 
    
    cout << "Time taken by optimize2: " << time_taken/1000.0 << " msec" << endl; 

    cout << endl << "***" << endl;
    for (int i = 0; i < gparams.all_op.size(); i++) {
        cout << gparams_2.all_op[i]->name << "  --  " << gparams_2.all_op[i]->U0.squaredNorm() << "  --  " << 
        gparams_2.all_op[i]->Y0.squaredNorm() << "  --  " << gparams_2.all_op[i]->weight.transpose() << endl;
    }
    cout << "***" << endl << endl;

    
    cout << endl << "-----------------------------------------------------";
    cout << endl << "------------ dt 50 warm Run -------------" << endl;
    cout << "-----------------------------------------------------" << endl << endl;

    GroptParams gparams_3;
    simple_diff(gparams_3, dt2, T_90, T_180, T_readout, TE, gmax, smax, N_moments, moment_tol, false);

    #ifdef USE_CHRONO
        t_start = chrono::steady_clock::now();
    #endif

    VectorXd out_3;
    gparams_2.X0 = out_2;
    gparams_2.do_init = false;
    optimize(gparams_2, out_3);

    cout << "Final3 " << gparams_3.total_n_feval << "  " << gparams_3.all_obj[0]->current_obj << endl;

    #ifdef USE_CHRONO
        t_end = chrono::steady_clock::now(); 
        time_taken = chrono::duration_cast<chrono::microseconds>(t_end - t_start).count();
    #else
        time_taken = 0.0;
    #endif 
    
    cout << "Time taken by optimize3: " << time_taken/1000.0 << " msec" << endl; 

    cout << endl << "***" << endl;
    for (int i = 0; i < gparams.all_op.size(); i++) {
        cout << gparams_2.all_op[i]->name << "  --  " << gparams_2.all_op[i]->U0.squaredNorm() << "  --  " << 
        gparams_2.all_op[i]->Y0.squaredNorm() << "  --  " << gparams_2.all_op[i]->weight.transpose() << endl;
    }
    cout << "***" << endl << endl;

    cout << endl << "-----------------------------------------------------";
    cout << endl << "------------ dt 50 warm interp Run -------------" << endl;
    cout << "-----------------------------------------------------" << endl << endl;
    
    
    GroptParams gparams_4;
    simple_diff(gparams_4, dt2, T_90, T_180, T_readout, TE, gmax, smax, N_moments, moment_tol, false);

    #ifdef USE_CHRONO
        t_start = chrono::steady_clock::now();
    #endif

    VectorXd out_4;
    gparams_4.interp_from_gparams(gparams, out);
    // gparams_4.grw_start = 1;
    // gparams_4.d_obj_thresh = 1.0e-2;
    gparams_4.do_init = false;

    optimize(gparams_4, out_4);

    cout << "Final4 " << gparams_4.total_n_feval << "  " << gparams_4.all_obj[0]->current_obj << endl;

    #ifdef USE_CHRONO
        t_end = chrono::steady_clock::now(); 
        time_taken = chrono::duration_cast<chrono::microseconds>(t_end - t_start).count();
    #else
        time_taken = 0.0;
    #endif  
    
    cout << "Time taken by optimize4: " << time_taken/1000.0 << " msec" << endl; 
    
    cout << endl << "***" << endl;
    for (int i = 0; i < gparams.all_op.size(); i++) {
        cout << gparams_4.all_op[i]->name << "  --  " << gparams_4.all_op[i]->U0.squaredNorm() << "  --  " << 
        gparams_4.all_op[i]->Y0.squaredNorm() << "  --  " << gparams_4.all_op[i]->weight.transpose() << endl;
    }
    cout << "***" << endl << endl;
    
    cout << "Done python_wrapper_warmstart_v1!" << endl << endl;
}



void diff_duty_cycle(double dt, double T_90, double T_180, double T_readout, double TE, 
                     int N_moments, double gmax, double smax, double bval, double duty_cycle,
                     double **out0, double **out1, double **out2, int **outsize) 
{
    double moment_tol = 1.0e-4;
    
    GroptParams gparams;
    gparams.verbose = 2;
    gparams.N_iter = 5000;
    gparams.cg_resid_tol = 1.0e-3;
    
    duty_diff(gparams, dt, bval, T_90, T_180, T_readout, TE, gmax, smax, N_moments, moment_tol, false, duty_cycle);

    VectorXd out;
    optimize(gparams, out);

    cout << "Final " << gparams.total_n_feval << endl;

    int N_out0 = out.size();
    *out0 = new double[N_out0];
    for(int i = 0; i < N_out0; i++) {
        out0[0][i] = out(i);
    }

    int N_out1 = 16;
    *out1 = new double[N_out1];
    out1[0][0] = gparams.final_good;
    out1[0][1] = gparams.last_iiter;
    out1[0][2] = gparams.total_n_feval;

    *outsize = new int[3];
    outsize[0][0] = N_out0;
    outsize[0][1] = N_out1;
    // outsize[0][2] = N_out2;
}


void spect_phase_contrast(double dt, double T, double gmax, double smax, double M0, double M1, int Neddy,
                          int Nset, double *in_setvals, double es_weight, int verbose, int start_ind1,
                          double **out0, double **out1, double **out2, int **outsize) 
{
    cout << "Starting" << endl;
    
    double moment_tol = 1.0e-4;
    
    int N = (int)(T/dt) + 1;

    VectorXd inv_vec;
    inv_vec.setOnes(N);

    VectorXd set_vals;
    set_vals.setOnes(N);
    set_vals.array() *= -9999999.0;
    set_vals(0) = 0.0;
    set_vals(N-1) = 0.0;

    if (Nset > 1) {
        if (N != Nset) {
            cout << "ERROR: Nset and N are not equal " << Nset << "  " << N << endl;
        }
        for(int i = 0; i < N; i++) {
            set_vals(i) = in_setvals[i];
        }
    }

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
    moments << 0, 0, 0, start_ind1, 0, M0, moment_tol,
               0, 1, 0, start_ind1, 0, M1, moment_tol;
    opM->set_params(moments);

    Op_Slew *opS = new Op_Slew(N, Naxis, dt);
    Op_Gradient *opG = new Op_Gradient(N, Naxis, dt);
    opS->set_params(smax);
    opG->set_params(gmax, set_vals);
    
    vector<Operator*> all_op;
    all_op.push_back(opG);
    all_op.push_back(opS);
    all_op.push_back(opM);

    vector<Operator*> all_obj;
    
    Op_Duty *opD = new Op_Duty(N, Naxis, dt);
    opD->weight(0) = 1.0e-4;
    all_obj.push_back(opD);

    Op_EddySpect *opES;
    if (Neddy > 0) {
        opES = new Op_EddySpect(N, Naxis, dt, Neddy);
        opES->weight(0) = es_weight;
        all_obj.push_back(opES);
    }


    VectorXd X;
    X.setOnes(N);
    X.array() *= gmax/10.0;
    for(int i = 0; i < N; i++) {
        if (set_vals(i) > -10000) {
            X(i) = set_vals(i);
        }
    }

    GroptParams gparams; 
    gparams.N = N;
    gparams.X0 = X;
    gparams.all_op = all_op; 
    gparams.all_obj = all_obj;
    gparams.inv_vec = inv_vec;
    gparams.fixer = fixer;
    gparams.set_vals = set_vals;
    gparams.verbose = verbose;

    gparams.N_iter = 5000;
    gparams.N_feval = 500000;
    gparams.cg_resid_tol = 1.0e-3;
    gparams.d_obj_thresh = 1.0e-3;

    gparams.set_vecs();
    gparams.update_vals();



    VectorXd out;
    optimize(gparams, out);

    cout << "Final " << gparams.total_n_feval << endl;

    int N_out0 = out.size();
    *out0 = new double[N_out0];
    for(int i = 0; i < N_out0; i++) {
        out0[0][i] = out(i);
    }

    int N_out1 = 16;
    *out1 = new double[N_out1];
    out1[0][0] = gparams.final_good;
    out1[0][1] = gparams.last_iiter;
    out1[0][2] = gparams.total_n_feval;

    int N_out2;
    if (Neddy > 0) {
        opES->Ax_temp.setZero();
        opES->forward(out, opES->Ax_temp, false, 0, true);

        cout << "opES->spec_norm2(0) = " << opES->spec_norm2(0) << endl;
        cout << "opES end norm = " <<  opES->Ax_temp.squaredNorm() << endl;
        
        N_out2 = opES->Ax_temp.size();
        *out2 = new double[N_out2];
        for(int i = 0; i < N_out2; i++) {
            out2[0][i] = opES->Ax_temp(i);
        }
    } else {
        N_out2 = 1;
        *out2 = new double[N_out2];
        out2[0][0] = 0.0;
    }

    *outsize = new int[3];
    outsize[0][0] = N_out0;
    outsize[0][1] = N_out1;
    outsize[0][2] = N_out2;
}




void spect_phase_contrast_spoiler(double dt, double T, double gmax, double smax, int Neddy,
                            double M0_spoil, double M0_ss, double M1_ss,
                            int ind_spoil_stop, int ind_ss_start0, int ind_ss_start1,
                            int eddy_start, int eddy_end,
                          int Nset, double *in_setvals, double es_weight, int verbose, 
                          double **out0, double **out1, double **out2, int **outsize) 
{
    cout << "Starting" << endl;
    
    double moment_tol = 1.0e-4;
    
    int N = (int)(T/dt) + 1;

    VectorXd inv_vec;
    inv_vec.setOnes(N);

    VectorXd set_vals;
    set_vals.setOnes(N);
    set_vals.array() *= -9999999.0;
    set_vals(0) = 0.0;
    set_vals(N-1) = 0.0;

    if (Nset > 1) {
        if (N != Nset) {
            cout << "ERROR: Nset and N are not equal " << Nset << "  " << N << endl;
        }
        for(int i = 0; i < N; i++) {
            set_vals(i) = in_setvals[i];
        }
    }

    VectorXd fixer;
    fixer.setOnes(N);
    for(int i = 0; i < N; i++) {
        if (set_vals(i) > -10000) {
            fixer(i) = 0.0;
        }
    }

    int Naxis = 1;
    int N_moments = 3;

    Op_Moments *opM = new Op_Moments(N, Naxis, dt, N_moments);
    MatrixXd moments(N_moments,7);
    moments << 0, 0, 0, 0, ind_spoil_stop, M0_spoil, moment_tol,
               0, 0, 0, ind_ss_start0, 0, M0_ss, moment_tol,
               0, 1, ind_ss_start1, ind_ss_start1, 0, M1_ss, moment_tol;
    opM->set_params(moments);

    Op_Slew *opS = new Op_Slew(N, Naxis, dt);
    Op_Gradient *opG = new Op_Gradient(N, Naxis, dt);
    opS->set_params(smax);
    opG->set_params(gmax, set_vals);
    
    vector<Operator*> all_op;
    all_op.push_back(opG);
    all_op.push_back(opS);
    all_op.push_back(opM);

    vector<Operator*> all_obj;
    
    Op_EddySpect2 *opES;
    if (Neddy > 0) {
        opES = new Op_EddySpect2(N, Naxis, dt, Neddy, eddy_start, eddy_end);
        opES->weight(0) = es_weight;
        all_obj.push_back(opES);
    }

    Op_Duty *opD = new Op_Duty(N, Naxis, dt);
    opD->weight(0) = 1.0e-6;
    all_obj.push_back(opD);


    VectorXd X;
    X.setOnes(N);
    X.array() *= gmax/10.0;
    for(int i = 0; i < N; i++) {
        if (set_vals(i) > -10000) {
            X(i) = set_vals(i);
        }
    }

    GroptParams gparams; 
    gparams.N = N;
    gparams.X0 = X;
    gparams.all_op = all_op; 
    gparams.all_obj = all_obj;
    gparams.inv_vec = inv_vec;
    gparams.fixer = fixer;
    gparams.set_vals = set_vals;
    gparams.verbose = verbose;

    gparams.N_iter = 5000;
    gparams.N_feval = 500000;
    gparams.cg_resid_tol = 1.0e-3;
    gparams.d_obj_thresh = 1.0e-3;

    gparams.set_vecs();
    gparams.update_vals();



    VectorXd out;
    optimize(gparams, out);

    cout << "Final " << gparams.total_n_feval << endl;

    int N_out0 = out.size();
    *out0 = new double[N_out0];
    for(int i = 0; i < N_out0; i++) {
        out0[0][i] = out(i);
    }

    int N_out1 = 16;
    *out1 = new double[N_out1];
    out1[0][0] = gparams.final_good;
    out1[0][1] = gparams.last_iiter;
    out1[0][2] = gparams.total_n_feval;

    int N_out2;
    if (Neddy > 0) {
        opES->Ax_temp.setZero();
        opES->forward(out, opES->Ax_temp, false, 0, true);

        cout << "opES->spec_norm2(0) = " << opES->spec_norm2(0) << endl;
        cout << "opES end norm = " <<  opES->Ax_temp.squaredNorm() << endl;
        
        N_out2 = opES->Ax_temp.size();
        *out2 = new double[N_out2];
        for(int i = 0; i < N_out2; i++) {
            out2[0][i] = opES->Ax_temp(i);
        }
    } else {
        N_out2 = 1;
        *out2 = new double[N_out2];
        out2[0][0] = 0.0;
    }

    *outsize = new int[3];
    outsize[0][0] = N_out0;
    outsize[0][1] = N_out1;
    outsize[0][2] = N_out2;
}


void simple_bipolar(double dt, double T, double gmax, double smax, double M0, double M1,
                   double l2_weight, int verbose,
                   double **out0, double **out1, double **out2, int **outsize) 
{   
    
    int N = (int)(T/dt) + 1;
    int Naxis = 1;

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



    int N_moments = 2;
    Op_Moments *opM = new Op_Moments(N, Naxis, dt, N_moments);
    MatrixXd moments(N_moments,7);

    moments << 0, 0, 0, 0, 0, M0, 1e-6,
                0, 1, 0, 0, 0, M1, 1e-6;

    opM->set_params(moments);



    Op_Slew *opS = new Op_Slew(N, Naxis, dt);
    Op_Gradient *opG = new Op_Gradient(N, Naxis, dt);
    Op_Duty *opD = new Op_Duty(N, Naxis, dt);
    
    opS->set_params(smax);
    opG->set_params(gmax, set_vals);
    
    opD->weight(0) = l2_weight;
    
    vector<Operator*> all_op;
    all_op.push_back(opG);
    all_op.push_back(opS);
    all_op.push_back(opM);

    vector<Operator*> all_obj;
    if (l2_weight > 0) {
        all_obj.push_back(opD);
    }


    VectorXd X;
    X.setOnes(Naxis*N);
    X.array() *= gmax/10.0;

    for(int i = 0; i < Naxis*N; i++) {
        if (set_vals(i) > -10000) {
            X(i) = set_vals(i);
        }
    }


    GroptParams gparams; 
    gparams.N = N;
    gparams.Naxis = Naxis;
    gparams.X0 = X;
    gparams.all_op = all_op; 
    gparams.all_obj = all_obj;
    gparams.inv_vec = inv_vec;
    gparams.fixer = fixer;
    gparams.set_vals = set_vals;
    gparams.verbose = verbose;

    gparams.N_iter = 500;
    gparams.N_feval = 10000;
    gparams.cg_resid_tol = 1.0e-3;
    gparams.d_obj_thresh = 1.0e-3;

    gparams.set_vecs();
    gparams.update_vals();


    VectorXd out;
    optimize(gparams, out);

    if (verbose > 0) {
        cout << "Done!  Final n_feval = " << gparams.total_n_feval << endl;
    }

    opM->Ax_temp.setZero();
    opM->forward(out, opM->Ax_temp, false, 0, true);

    if (verbose > 0) {
        cout << "opM =  " << opM->Ax_temp.transpose() << endl;
    }
    
    int N_out0 = out.size();
    *out0 = new double[N_out0];
    for(int i = 0; i < N_out0; i++) {
        out0[0][i] = out(i);
    }

    int N_out1 = 16;
    *out1 = new double[N_out1];
    out1[0][0] = gparams.final_good;
    out1[0][1] = gparams.last_iiter;
    out1[0][2] = gparams.total_n_feval;


    int N_out2 = 16;
    *out2 = new double[N_out2];
    out2[0][0] = gparams.final_good;
    out2[0][1] = gparams.last_iiter;
    out2[0][2] = gparams.total_n_feval;


    *outsize = new int[3];
    outsize[0][0] = N_out0;
    outsize[0][1] = N_out1;
    outsize[0][2] = N_out2;

}



void spect_phase_contrast_spoiler2(double dt, double T, double gmax, double smax, int Neddy,
                            double M0_spoil, double M0_ss, double M1_ss,
                            int ind_spoil_start, int ind_spoil_stop, int ind_ss_start0, int ind_ss_start1,
                            int eddy_start, int eddy_end,
                            int Nset, double *in_setvals, double es_weight, double l2_weight, double *in_spect, int cg_iter, int verbose, 
                            double **out0, double **out1, double **out2, int **outsize) 
{
    double moment_tol = 1.0e-4;
    
    int N = (int)(T/dt) + 1;

    VectorXd inv_vec;
    inv_vec.setOnes(N);

    VectorXd set_vals;
    set_vals.setOnes(N);
    set_vals.array() *= -9999999.0;
    set_vals(0) = 0.0;
    set_vals(N-1) = 0.0;

    if (Nset > 1) {
        if (N != Nset) {
            cout << "ERROR: Nset and N are not equal " << Nset << "  " << N << endl;
        }
        for(int i = 0; i < N; i++) {
            set_vals(i) = in_setvals[i];
        }
    }

    VectorXd fixer;
    fixer.setOnes(N);
    for(int i = 0; i < N; i++) {
        if (set_vals(i) > -10000) {
            fixer(i) = 0.0;
        }
    }

    int Naxis = 1;

    MatrixXd moments;
    int N_moments = 3;
    if (M0_spoil >= 0) {
        N_moments = 3;
        moments.setZero(N_moments,7);
        moments << 0, 0, 0, ind_spoil_start, ind_spoil_stop, 2*M0_spoil, M0_spoil+moment_tol,
                0, 0, 0, ind_ss_start0, 0, M0_ss, moment_tol,
                0, 1, ind_ss_start1, ind_ss_start1, 0, M1_ss, moment_tol;
    } else {
        N_moments = 2;
        moments.setZero(N_moments,7);
        moments << 0, 0, 0, ind_ss_start0, 0, M0_ss, moment_tol,
                0, 1, ind_ss_start1, ind_ss_start1, 0, M1_ss, moment_tol;
    }
    Op_Moments *opM = new Op_Moments(N, Naxis, dt, N_moments);
    opM->set_params(moments);

    Op_Slew *opS = new Op_Slew(N, Naxis, dt);
    Op_Gradient *opG = new Op_Gradient(N, Naxis, dt);
    opS->set_params(smax);
    opG->set_params(gmax, set_vals);
    
    vector<Operator*> all_op;
    all_op.push_back(opG);
    all_op.push_back(opS);
    all_op.push_back(opM);

    vector<Operator*> all_obj;

    Op_EddySpect2 *opES;
    if (Neddy > 0) {
        opES = new Op_EddySpect2(N, Naxis, dt, Neddy, eddy_start, eddy_end);
        opES->weight(0) = es_weight;
        opES->set_spect(in_spect);
        all_obj.push_back(opES);
    }

    if (l2_weight > 0) {
        Op_Duty *opD = new Op_Duty(N, Naxis, dt);
        opD->weight(0) = l2_weight;
        all_obj.push_back(opD);
    }
    
   

    


    VectorXd X;
    X.setOnes(N);
    X.array() *= gmax/10.0;
    for(int i = 0; i < N; i++) {
        if (set_vals(i) > -10000) {
            X(i) = set_vals(i);
        }
    }

    GroptParams gparams; 
    gparams.N = N;
    gparams.X0 = X;
    gparams.all_op = all_op; 
    gparams.all_obj = all_obj;
    gparams.inv_vec = inv_vec;
    gparams.fixer = fixer;
    gparams.set_vals = set_vals;
    gparams.verbose = verbose;

    gparams.N_iter = 1000;
    gparams.N_feval = 50000;
    gparams.cg_resid_tol = 1.0e-3;
    gparams.d_obj_thresh = 1.0e-3;
    gparams.cg_niter = cg_iter;
    gparams.verbose_int = 50;

    gparams.set_vecs();
    gparams.update_vals();



    VectorXd out;
    optimize(gparams, out);

    // cout << "Final " << gparams.total_n_feval << endl;

    // opM->Ax_temp.setZero();
    // opM->forward(out, opM->Ax_temp, false, 0, true);

    // cout << "opM =  " << opM->Ax_temp.transpose() << endl;

    int N_out0 = out.size();
    *out0 = new double[N_out0];
    for(int i = 0; i < N_out0; i++) {
        out0[0][i] = out(i);
    }

    int N_out1 = 16;
    *out1 = new double[N_out1];
    out1[0][0] = gparams.final_good;
    out1[0][1] = gparams.last_iiter;
    out1[0][2] = gparams.total_n_feval;

    int N_out2;
    if (Neddy > 0) {
        opES->Ax_temp.setZero();
        opES->forward(out, opES->Ax_temp, false, 0, true);

        // cout << "opES->spec_norm2(0) = " << opES->spec_norm2(0) << endl;
        // cout << "opES end norm = " <<  opES->Ax_temp.squaredNorm() << endl;
        
        N_out2 = opES->Ax_temp.size();
        *out2 = new double[N_out2];
        for(int i = 0; i < N_out2; i++) {
            out2[0][i] = opES->Ax_temp(i);
        }
    } else {
        N_out2 = 1;
        *out2 = new double[N_out2];
        out2[0][0] = 0.0;
    }

    *outsize = new int[3];
    outsize[0][0] = N_out0;
    outsize[0][1] = N_out1;
    outsize[0][2] = N_out2;
}

void diff_pre_eddy(double dt, double T_90, double T_180, double T_readout, double T_pre, double TE, 
                   int moment_order, double gmax, double smax, double *eddy_lam_in, int Nlam, double maxwell_tol, double b_weight,
                   double **out0, double **out1, double **out2, int **outsize)
{
    cout << "diff_pre_eddy:" << endl << endl;

    int Naxis = 1;
    int N_pre = (int)(T_pre/dt);
    cout << "N_pre:" << N_pre << endl;

    int N = (int)((TE-T_readout)/dt) + N_pre + 1;

    // ---- Inversion setup
    int ind_inv = (int)(TE/2.0/dt) + N_pre;

    VectorXd inv_vec;
    inv_vec.setOnes(Naxis*N);

    for (int j = 0; j < Naxis; j++) {
        for(int i = j*N+ind_inv; i < (j+1)*N; i++) {
            inv_vec(i) = -1.0;
        }
    }


    // ---- Fixed gradient value setter
    int ind_90_end, ind_180_start, ind_180_end;

    ind_90_end = ceil(T_90/dt);
    ind_180_start = floor((TE/2.0 - T_180/2.0)/dt);
    ind_180_end = ceil((TE/2.0 + T_180/2.0)/dt);

    VectorXd set_vals;
    set_vals.setOnes(Naxis*N);
    set_vals.array() *= -9999999.0;

    for (int j = 0; j < Naxis; j++) {
        for(int i = j*N+N_pre; i <= j*N+ind_90_end+N_pre; i++) {
            set_vals(i) = 0.0;
        }
        for(int i = j*N+ind_180_start+N_pre; i <= j*N+ind_180_end+N_pre; i++) {
            set_vals(i) = 0.0;
        }
        set_vals(j*N+0) = 0.0;
        set_vals(j*N+N-1) = 0.0;
    }


    VectorXd fixer;
    fixer.setOnes(Naxis*N);
    for(int i = 0; i < Naxis*N; i++) {
        if (set_vals(i) > -10000) {
            fixer(i) = 0.0;
        }
    }

    int N_moments = (moment_order+1) * Naxis;
    Op_Moments *opM = new Op_Moments(N, Naxis, dt, N_moments);
    MatrixXd moments(N_moments,7);
    moments.setZero(N_moments, 7);
    int ii = 0;
    for (int j = 0; j < Naxis; j++) {
        for (int i = 0; i <= moment_order; i++) {
            moments(ii, 0) = j;
            moments(ii, 1) = i;
            moments(ii, 3) = N_pre;
            moments(ii, 6) = 1e-4;
            ii += 1;
        }
    }
    opM->set_params(moments);
    cout << "Moment params:" << endl << moments << endl;

    VectorXd eddy_lam = Map<VectorXd>(eddy_lam_in,Nlam);

    Op_Slew *opS = new Op_Slew(N, Naxis, dt);
    Op_Gradient *opG = new Op_Gradient(N, Naxis, dt);
    Op_BVal *opB = new Op_BVal(N, Naxis, dt);
    Op_Eddy *opE = new Op_Eddy(N, Naxis, dt, Nlam);
    Op_Maxwell *opMax = new Op_Maxwell(N, Naxis, dt);
    

    opS->set_params(smax);
    opG->set_params(gmax, set_vals);
    opB->set_start(N_pre);
    opE->prep_A(eddy_lam, 1.0e-4);
    opMax->set_params(maxwell_tol, N_pre, ind_inv, N-1);
    opB->weight(0) = b_weight;

    vector<Operator*> all_op;
    all_op.push_back(opG);
    all_op.push_back(opS);
    all_op.push_back(opM);
    if (eddy_lam(0) > 0) { all_op.push_back(opE); }
    if (maxwell_tol > 0) { all_op.push_back(opMax); }

    vector<Operator*> all_obj;
    all_obj.push_back(opB);

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
    gparams.verbose = 3;

    gparams.set_vecs();
    gparams.defaults_diffusion();
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
    // outsize[0][1] = N_out1;
    // outsize[0][2] = N_out2;
    
    cout << "Done diff_pre_eddy! " << N_out0 << endl << endl;

    opB->Ax_temp.setZero();
    opB->forward(out, opB->Ax_temp, false, 0, true);

    cout << "final b-val = " << opB->Ax_temp.squaredNorm() << endl;

    // cout << endl << out.transpose() << endl;

    cout << endl << "eddy_lam = " << eddy_lam << endl;
    
    return; 
}


// This is a call for general purpose testing, i.e. for testing the optimization hyperparameters
void threed_diff(double *params0, double *params1, double **out0, double **out1, double **out2, int **outsize) 
{
    cout << "threed_diff:" << endl << endl;
    
    double dt = params0[0];
    double T_90 = params0[1];
    double T_180 = params0[2];
    double T_readout = params0[3];
    double TE = params0[4];

    int MMT = params0[5];
    double moment_tol = params0[6];

    double gmax = params0[7];
    double smax = params0[8];


    int N = (int)((TE-T_readout)/dt) + 1;
    int ind_inv = (int)(TE/2.0/dt);
    int Naxis = 3;

    VectorXd inv_vec;
    inv_vec.setOnes(Naxis*N);

    for (int j = 0; j < Naxis; j++) {
        for(int i = j*N+ind_inv; i < (j+1)*N; i++) {
            inv_vec(i) = -1.0;
        }
    }


    int ind_90_end, ind_180_start, ind_180_end;

    ind_90_end = ceil(T_90/dt);
    ind_180_start = floor((TE/2.0 - T_180/2.0)/dt);
    ind_180_end = ceil((TE/2.0 + T_180/2.0)/dt);

    VectorXd set_vals;
    set_vals.setOnes(Naxis*N);
    set_vals.array() *= -9999999.0;

    for (int j = 0; j < Naxis; j++) {
        cout << j*N+0 << " " << j*N+ind_90_end << " " << j*N+ind_180_start << " " << j*N+ind_180_end << endl;
        for(int i = j*N+0; i <= j*N+ind_90_end; i++) {
            set_vals(i) = 0.0;
        }
        for(int i = j*N+ind_180_start; i <= j*N+ind_180_end; i++) {
            set_vals(i) = 0.0;
        }
        set_vals(j*N+0) = 0.0;
        set_vals(j*N+N-1) = 0.0;
    }


    VectorXd fixer;
    fixer.setOnes(Naxis*N);
    for(int i = 0; i < Naxis*N; i++) {
        if (set_vals(i) > -10000) {
            fixer(i) = 0.0;
        }
    }

    int N_moments = (MMT+1)* Naxis;
    Op_Moments *opM = new Op_Moments(N, Naxis, dt, N_moments);
    MatrixXd moments(N_moments,7);
    moments.setZero(N_moments, 7);
    int ii = 0;
    for (int j = 0; j < Naxis; j++) {
        for (int i = 0; i <= MMT; i++) {
            moments(ii, 0) = j;
            moments(ii, 1) = i;
            moments(ii, 6) = moment_tol;
            ii += 1;
        }
    }
    opM->set_params(moments);
    cout << "Moment params:" << endl << moments << endl;

    Op_Slew *opS = new Op_Slew(N, Naxis, dt);
    Op_Gradient *opG = new Op_Gradient(N, Naxis, dt);
    Op_Duty *opD = new Op_Duty(N, Naxis, dt);
    Op_BTensor *opBT = new Op_BTensor(N, Naxis, dt);
    
    opS->set_params(smax);
    opG->set_params(gmax, set_vals);
    opBT->set_params(params0[10],params0[11],params0[12]);
    
    double l2_reg = params0[9];
    opD->weight(0) = l2_reg;

    vector<Operator*> all_op;
    all_op.push_back(opG);
    all_op.push_back(opS);
    all_op.push_back(opM);
    all_op.push_back(opBT);

    vector<Operator*> all_obj;
    if (l2_reg > 0) {
        all_obj.push_back(opD);
    }

    VectorXd X;
    X.setOnes(Naxis*N);
    X.array() *= inv_vec.array() * fixer.array() * gmax/10.0;

    GroptParams gparams;
    gparams.N = N;
    gparams.Naxis = Naxis;
    gparams.X0 = X;
    gparams.all_op = all_op; 
    gparams.all_obj = all_obj;
    gparams.inv_vec = inv_vec;
    gparams.fixer = fixer;
    gparams.set_vals = set_vals;
    gparams.verbose = params0[13];
    
    gparams.N_iter = 3000;
    gparams.N_feval = 50000;
    gparams.d_obj_thresh = 1e-3;

    gparams.set_vecs();
    gparams.update_vals();

    VectorXd out;
    optimize(gparams, out);

    cout << "  !!!!!!!!!  " << " -- Done -- " << "  !!!!!!!!!  " << endl << endl;

    opBT->test(out);

    int N_out0 = out.size();
    *out0 = new double[N_out0];
    for(int i = 0; i < N_out0; i++) {
        out0[0][i] = out(i);
    }

    // int N_out1 = 16;
    // *out1 = new double[N_out1];
    // out1[0][0] = gparams.total_n_feval;
    // out1[0][1] = gparams.all_obj[0]->current_obj;
    // out1[0][2] = time_taken1 * 1e-6;
    // out1[0][3] = gparams.last_iiter;

    // int N_out2 = 160;
    // VectorXd v_out2;
    // v_out2.setOnes(N_out2);

    // interp_vec2vec(out, v_out2);

    // *out2 = new double[N_out2];
    // for(int i = 0; i < N_out2; i++) {
    //     out2[0][i] = v_out2(i);
    // }


    *outsize = new int[3];
    outsize[0][0] = N_out0;
    // outsize[0][1] = N_out1;
    // outsize[0][2] = N_out2;
    

    cout << "Done threed_diff!" << endl << endl;
}


void rewinder3(double dt, double T, double gmax, double smax, double *G0_in, double *M0_in,
               double l2_weight,
               double **out0, double **out1, double **out2, int **outsize) 
{

    int N = (int)(T/dt) + 1;
    int Naxis = 3;

    VectorXd inv_vec;
    inv_vec.setOnes(Naxis*N);

    VectorXd set_vals;
    set_vals.setOnes(Naxis*N);
    set_vals.array() *= -9999999.0;

    for (int j = 0; j < Naxis; j++) {
        set_vals((j*N)) = G0_in[j];
        set_vals((j*N)+(N-1)) = 0.0;
    }

    VectorXd fixer;
    fixer.setOnes(Naxis*N);
    for(int i = 0; i < Naxis*N; i++) {
        if (set_vals(i) > -10000) {
            fixer(i) = 0.0;
        }
    }


    int N_moments = 3;
    Op_Moments *opM = new Op_Moments(N, Naxis, dt, N_moments);
    MatrixXd moments(N_moments,7);
    moments << 0, 0, 0, 0, 0, M0_in[0], 1e-6,
               1, 0, 0, 0, 0, M0_in[1], 1e-6,
               2, 0, 0, 0, 0, M0_in[2], 1e-6,
    opM->set_params(moments);

    Op_Slew *opS = new Op_Slew(N, Naxis, dt);
    Op_Gradient *opG = new Op_Gradient(N, Naxis, dt);
    Op_Duty *opD = new Op_Duty(N, Naxis, dt);
    
    opS->set_params(smax);
    opG->set_params(gmax, set_vals);
    opD->weight(0) = l2_weight;
    
    vector<Operator*> all_op;
    all_op.push_back(opG);
    all_op.push_back(opS);
    all_op.push_back(opM);

    vector<Operator*> all_obj;
    all_obj.push_back(opD);

    VectorXd X;
    X.setOnes(Naxis*N);
    X.array() *= gmax/10.0;

    for(int i = 0; i < Naxis*N; i++) {
        if (set_vals(i) > -10000) {
            X(i) = set_vals(i);
        }
    }


    GroptParams gparams; 
    gparams.N = N;
    gparams.Naxis = Naxis;
    gparams.X0 = X;
    gparams.all_op = all_op; 
    gparams.all_obj = all_obj;
    gparams.inv_vec = inv_vec;
    gparams.fixer = fixer;
    gparams.set_vals = set_vals;

    gparams.N_iter = 2000;
    gparams.N_feval = 50000;
    gparams.cg_resid_tol = 1.0e-3;
    gparams.d_obj_thresh = 1.0e-3;

    gparams.set_vecs();
    gparams.update_vals();


    VectorXd out;
    optimize(gparams, out);

    cout << "Done!  Final n_feval = " << gparams.total_n_feval << endl;

    opM->Ax_temp.setZero();
    opM->forward(out, opM->Ax_temp, false, 0, true);

    cout << "opM =  " << opM->Ax_temp.transpose() << endl;

    
    int N_out0 = out.size();
    *out0 = new double[N_out0];
    for(int i = 0; i < N_out0; i++) {
        out0[0][i] = out(i);
    }

    int N_out1 = 16;
    *out1 = new double[N_out1];
    out1[0][0] = gparams.final_good;
    out1[0][1] = gparams.last_iiter;
    out1[0][2] = gparams.total_n_feval;

    // int N_out1 = 16;
    // *out1 = new double[N_out1];
    // out1[0][0] = gparams.total_n_feval;
    // out1[0][1] = gparams.all_obj[0]->current_obj;
    // out1[0][2] = time_taken1 * 1e-6;
    // out1[0][3] = gparams.last_iiter;

    // int N_out2 = 160;
    // VectorXd v_out2;
    // v_out2.setOnes(N_out2);

    // interp_vec2vec(out, v_out2);

    // *out2 = new double[N_out2];
    // for(int i = 0; i < N_out2; i++) {
    //     out2[0][i] = v_out2(i);
    // }


    *outsize = new int[3];
    outsize[0][0] = N_out0;
    outsize[0][1] = N_out1;
    // outsize[0][2] = N_out2;


    for (int i = 0; i < gparams.all_op.size(); i++) {
        VectorXd temp;
        temp.setZero(all_op[i]->Y0.size());
        gparams.all_op[i]->forward(out, temp, false, 0, true);

        cout << "***** " << gparams.all_op[i]->name << " *****" << endl
        << "  --  Checks: " << gparams.all_op[i]->hist_check.col(gparams.last_iiter).transpose() << endl
        // << "  --  Vals: " << temp.transpose() << endl
        << endl;
    }
    

    cout << "Done rewinder3!" << endl << endl;

}


void acoustic_v1(double dt, double T, double gmax, double smax, double *G0_in, double *H_in,
                double l2_weight, double a_weight, int verbose,
                double **out0, double **out1, double **out2, int **outsize) 
{
    VectorXcd H_load;
    H_load.setOnes(15002);
    
    ifstream f_in; 
    complex<double> temp;
    f_in.open("mean_H_v1.bin", ios::binary);
    for (int i=0; i<7501; i++) {
        f_in.read(reinterpret_cast<char*>(&temp), sizeof(complex<double>));
        H_load(i) = temp;
    } 
    f_in.close();

    cout << endl << "H_load:" << endl;
    cout << H_load.head(10) << endl << endl;
    
    
    int N = (int)(T/dt) + 1;
    int Naxis = 1;

    VectorXd inv_vec;
    inv_vec.setOnes(Naxis*N);

    VectorXd set_vals;
    set_vals.setOnes(Naxis*N);
    set_vals.array() *= -9999999.0;

    VectorXd fixer;
    fixer.setOnes(Naxis*N);


    for(int i = 0; i < Naxis*N; i++) {
        if (G0_in[i] > -10000) {
            set_vals(i) = G0_in[i];
            fixer(i) = 0.0;
        }
    }



    int N_moments = 4;
    Op_Moments *opM = new Op_Moments(N, Naxis, dt, N_moments);
    MatrixXd moments(N_moments,7);
    moments << 0, 0, 0, 0, N, 0.0, 1e-6,
               0, 1, 0, 0, N, 0.0, 1e-6,
               0, 0, 0, 0, N/2, 0.0, 1e-6,
               0, 1, 0, 0, N/2, 0.0, 1e-6,
    opM->set_params(moments);

    Op_Slew *opS = new Op_Slew(N, Naxis, dt);
    Op_Gradient *opG = new Op_Gradient(N, Naxis, dt);
    Op_Duty *opD = new Op_Duty(N, Naxis, dt);
    Op_Acoustic *opA = new Op_Acoustic(N, Naxis, dt, H_load);
    
    opS->set_params(smax);
    opG->set_params(gmax, set_vals);
    
    opD->weight(0) = l2_weight;
    opA->weight(0) = a_weight;
    // opA->set_fixer(fixer);
    // opA->set_inv_vec(inv_vec);
    
    vector<Operator*> all_op;
    all_op.push_back(opG);
    all_op.push_back(opS);
    all_op.push_back(opM);

    vector<Operator*> all_obj;
    if (l2_weight > 0) {
        all_obj.push_back(opD);
    } else {
        all_obj.push_back(opA);
    }

    VectorXd X;
    X.setOnes(Naxis*N);
    X.array() *= gmax/10.0;

    for(int i = 0; i < Naxis*N; i++) {
        if (set_vals(i) > -10000) {
            X(i) = set_vals(i);
        }
    }


    GroptParams gparams; 
    gparams.N = N;
    gparams.Naxis = Naxis;
    gparams.X0 = X;
    gparams.all_op = all_op; 
    gparams.all_obj = all_obj;
    gparams.inv_vec = inv_vec;
    gparams.fixer = fixer;
    gparams.set_vals = set_vals;
    gparams.verbose = verbose;

    gparams.N_iter = 400;
    gparams.N_feval = 5000;
    gparams.cg_resid_tol = 1.0e-3;
    gparams.d_obj_thresh = 1.0e-3;

    gparams.set_vecs();
    gparams.update_vals();


    VectorXd out;
    optimize(gparams, out);

    cout << "Done!  Final n_feval = " << gparams.total_n_feval << endl;

    opM->Ax_temp.setZero();
    opM->forward(out, opM->Ax_temp, false, 0, true);

    cout << "opM =  " << opM->Ax_temp.transpose() << endl;

    
    int N_out0 = out.size();
    *out0 = new double[N_out0];
    for(int i = 0; i < N_out0; i++) {
        out0[0][i] = out(i);
    }

    int N_out1 = 16;
    *out1 = new double[N_out1];
    out1[0][0] = gparams.final_good;
    out1[0][1] = gparams.last_iiter;
    out1[0][2] = gparams.total_n_feval;


    
    VectorXd outvec2;
    outvec2.setZero(out.size());

    opA->add2AtAx(out, outvec2);


    int N_out2 = outvec2.size();
    *out2 = new double[N_out2];
    for(int i = 0; i < outvec2.size(); i++) {
        out2[0][i] = outvec2(i);
    }


    *outsize = new int[3];
    outsize[0][0] = N_out0;
    outsize[0][1] = N_out1;
    outsize[0][2] = N_out2;


    for (int i = 0; i < gparams.all_op.size(); i++) {
        VectorXd temp;
        temp.setZero(all_op[i]->Y0.size());
        gparams.all_op[i]->forward(out, temp, false, 0, true);

        cout << "***** " << gparams.all_op[i]->name << " *****" << endl
        << "  --  Checks: " << gparams.all_op[i]->hist_check.col(gparams.last_iiter).transpose() << endl
        // << "  --  Vals: " << temp.transpose() << endl
        << endl;
    }

    double final_GA;

    opA->Ax_temp.setZero();
    opA->forward(out, opA->Ax_temp, false, 0, true);
    final_GA = opA->Ax_temp.squaredNorm();

    cout << endl << "!!!!!! Final GA = " << final_GA << endl << endl;
     

    cout << "Done acoustic_v1!" << endl << endl;

}


void acoustic_v2(double dt, double T, double gmax, double smax, double *G0_in, complex<double> *H_in,
                double l2_weight, double a_weight, int verbose,
                double **out0, double **out1, double **out2, int **outsize) 
{
    VectorXcd H_load;
    H_load.setOnes(15002);
    
    for (int i=0; i<7501; i++) {
        H_load(i) = H_in[i];
    }

    cout << endl << "H_load:" << endl;
    cout << H_load.head(10) << endl << endl;
    
    
    int N = (int)(T/dt) + 1;
    int Naxis = 1;

    VectorXd inv_vec;
    inv_vec.setOnes(Naxis*N);

    VectorXd set_vals;
    set_vals.setOnes(Naxis*N);
    set_vals.array() *= -9999999.0;

    VectorXd fixer;
    fixer.setOnes(Naxis*N);


    for(int i = 0; i < Naxis*N; i++) {
        if (G0_in[i] > -10000) {
            set_vals(i) = G0_in[i];
            fixer(i) = 0.0;
        }
    }



    int N_moments = 4;
    Op_Moments *opM = new Op_Moments(N, Naxis, dt, N_moments);
    MatrixXd moments(N_moments,7);
    moments << 0, 0, 0, 0, N, 0.0, 1e-6,
               0, 1, 0, 0, N, 0.0, 1e-6,
               0, 0, 0, 0, N/2, 0.0, 1e-6,
               0, 1, 0, 0, N/2, 0.0, 1e-6,
    opM->set_params(moments);

    Op_Slew *opS = new Op_Slew(N, Naxis, dt);
    Op_Gradient *opG = new Op_Gradient(N, Naxis, dt);
    Op_Duty *opD = new Op_Duty(N, Naxis, dt);
    Op_Acoustic *opA = new Op_Acoustic(N, Naxis, dt, H_load);
    
    opS->set_params(smax);
    opG->set_params(gmax, set_vals);
    
    opD->weight(0) = l2_weight;
    opA->weight(0) = a_weight;
    // opA->set_fixer(fixer);
    // opA->set_inv_vec(inv_vec);
    
    vector<Operator*> all_op;
    all_op.push_back(opG);
    all_op.push_back(opS);
    all_op.push_back(opM);

    vector<Operator*> all_obj;
    if (l2_weight > 0) {
        all_obj.push_back(opD);
    } else {
        all_obj.push_back(opA);
    }

    VectorXd X;
    X.setOnes(Naxis*N);
    X.array() *= gmax/10.0;

    for(int i = 0; i < Naxis*N; i++) {
        if (set_vals(i) > -10000) {
            X(i) = set_vals(i);
        }
    }


    GroptParams gparams; 
    gparams.N = N;
    gparams.Naxis = Naxis;
    gparams.X0 = X;
    gparams.all_op = all_op; 
    gparams.all_obj = all_obj;
    gparams.inv_vec = inv_vec;
    gparams.fixer = fixer;
    gparams.set_vals = set_vals;
    gparams.verbose = verbose;


    gparams.N_iter = 1000;
    gparams.N_feval = 50000;
    gparams.cg_resid_tol = 1.0e-3;
    gparams.d_obj_thresh = 1.0e-3;
    gparams.cg_niter = 20;
    gparams.verbose_int = 50;

    gparams.set_vecs();
    gparams.update_vals();


    VectorXd out;
    optimize(gparams, out);

    cout << "Done!  Final n_feval = " << gparams.total_n_feval << endl;

    opM->Ax_temp.setZero();
    opM->forward(out, opM->Ax_temp, false, 0, true);

    cout << "opM =  " << opM->Ax_temp.transpose() << endl;

    
    int N_out0 = out.size();
    *out0 = new double[N_out0];
    for(int i = 0; i < N_out0; i++) {
        out0[0][i] = out(i);
    }

    int N_out1 = 16;
    *out1 = new double[N_out1];
    out1[0][0] = gparams.final_good;
    out1[0][1] = gparams.last_iiter;
    out1[0][2] = gparams.total_n_feval;


    
    // VectorXd outvec2;
    // outvec2.setZero(out.size());
    // opA->add2AtAx(out, outvec2);

    VectorXd outvec2;
    outvec2.setZero(opA->Y0.size());
    opA->forward(out, outvec2, false, 0, true);


    int N_out2 = outvec2.size();
    *out2 = new double[N_out2];
    for(int i = 0; i < outvec2.size(); i++) {
        out2[0][i] = outvec2(i);
    }


    *outsize = new int[3];
    outsize[0][0] = N_out0;
    outsize[0][1] = N_out1;
    outsize[0][2] = N_out2;


    for (int i = 0; i < gparams.all_op.size(); i++) {
        VectorXd temp;
        temp.setZero(all_op[i]->Y0.size());
        gparams.all_op[i]->forward(out, temp, false, 0, true);

        cout << "***** " << gparams.all_op[i]->name << " *****" << endl
        << "  --  Checks: " << gparams.all_op[i]->hist_check.col(gparams.last_iiter).transpose() << endl
        // << "  --  Vals: " << temp.transpose() << endl
        << endl;
    }

    double final_GA;

    opA->Ax_temp.setZero();
    opA->forward(out, opA->Ax_temp, false, 0, true);
    final_GA = opA->Ax_temp.squaredNorm();

    cout << endl << "!!!!!! Final GA = " << final_GA << endl << endl;
     

    cout << "Done acoustic_v2!" << endl << endl;

}



void acoustic_v3(double dt, double T, double gmax, double smax, double *G0_in, 
                complex<double> *H_in, int N_H,
                int N_moments, double *moments_params,
                double l2_weight, double a_weight, int verbose,
                int p_iter, int p_cg_iter, int p_obj_min, double p_obj_scale,
                double **out0, double **out1, double **out2, int **outsize) 
{
    VectorXcd H_load;
    H_load.setOnes(N_H);
    
    for (int i=0; i<N_H; i++) {
        H_load(i) = H_in[i];
    }

    cout << endl << "H_load:" << endl;
    cout << H_load.head(10) << endl << endl;
    
    
    int N = (int)(T/dt) + 1;
    int Naxis = 1;

    VectorXd inv_vec;
    inv_vec.setOnes(Naxis*N);

    VectorXd set_vals;
    set_vals.setOnes(Naxis*N);
    set_vals.array() *= -9999999.0;

    VectorXd fixer;
    fixer.setOnes(Naxis*N);


    for(int i = 0; i < Naxis*N; i++) {
        if (G0_in[i] > -10000) {
            set_vals(i) = G0_in[i];
            fixer(i) = 0.0;
        }
    }


    Op_Moments *opM = new Op_Moments(N, Naxis, dt, N_moments);
    MatrixXd moments;
    moments.setZero(N_moments,7);
    for (int i = 0; i < N_moments; i++) {
        for (int j = 0; j < 7; j++) {
            moments(i,j) = moments_params[j + i*7];
        }
    }
    opM->set_params(moments);

    Op_Slew *opS = new Op_Slew(N, Naxis, dt);
    Op_Gradient *opG = new Op_Gradient(N, Naxis, dt);
    Op_Duty *opD = new Op_Duty(N, Naxis, dt);
    Op_Acoustic *opA = new Op_Acoustic(N, Naxis, dt, H_load, N_H);
    
    opS->set_params(smax);
    opG->set_params(gmax, set_vals);
    
    opD->weight(0) = l2_weight;
    opA->weight(0) = a_weight;
    // opA->set_fixer(fixer);
    // opA->set_inv_vec(inv_vec);
    
    vector<Operator*> all_op;
    all_op.push_back(opG);
    all_op.push_back(opS);
    all_op.push_back(opM);

    vector<Operator*> all_obj;
    if (l2_weight > 0) {
        all_obj.push_back(opD);
    } 
    if (a_weight > 0) {
        all_obj.push_back(opA);
    }

    VectorXd X;
    X.setOnes(Naxis*N);
    X.array() *= gmax/10.0;

    for(int i = 0; i < Naxis*N; i++) {
        if (set_vals(i) > -10000) {
            X(i) = set_vals(i);
        }
    }


    GroptParams gparams; 
    gparams.N = N;
    gparams.Naxis = Naxis;
    gparams.X0 = X;
    gparams.all_op = all_op; 
    gparams.all_obj = all_obj;
    gparams.inv_vec = inv_vec;
    gparams.fixer = fixer;
    gparams.set_vals = set_vals;
    gparams.verbose = verbose;


    gparams.N_iter = p_iter;
    gparams.N_feval = 500000;
    gparams.cg_resid_tol = 1.0e-3;
    gparams.d_obj_thresh = 1.0e-3;
    gparams.cg_niter = p_cg_iter;
    gparams.verbose_int = 50;

    gparams.obj_min = p_obj_min;
    gparams.obj_scale = p_obj_scale;
    
    gparams.set_vecs();
    gparams.update_vals();


    VectorXd out;
    optimize(gparams, out);

    cout << "Done!  Final n_feval = " << gparams.total_n_feval << endl;

    opM->Ax_temp.setZero();
    opM->forward(out, opM->Ax_temp, false, 0, true);

    cout << "opM =  " << opM->Ax_temp.transpose() << endl;

    
    int N_out0 = out.size();
    *out0 = new double[N_out0];
    for(int i = 0; i < N_out0; i++) {
        out0[0][i] = out(i);
    }

    int N_out1 = 16;
    *out1 = new double[N_out1];
    out1[0][0] = gparams.final_good;
    out1[0][1] = gparams.last_iiter;
    out1[0][2] = gparams.total_n_feval;

    // VectorXd outvec2;
    // outvec2.setZero(out.size());
    // opA->add2AtAx(out, outvec2);

    VectorXd outvec2;
    outvec2.setZero(opA->Y0.size());
    opA->forward(out, outvec2, false, 0, true);


    int N_out2 = outvec2.size();
    *out2 = new double[N_out2];
    for(int i = 0; i < outvec2.size(); i++) {
        out2[0][i] = outvec2(i);
    }


    *outsize = new int[3];
    outsize[0][0] = N_out0;
    outsize[0][1] = N_out1;
    outsize[0][2] = N_out2;


    for (int i = 0; i < gparams.all_op.size(); i++) {
        VectorXd temp;
        temp.setZero(all_op[i]->Y0.size());
        gparams.all_op[i]->forward(out, temp, false, 0, true);

        cout << "***** " << gparams.all_op[i]->name << " *****" << endl
        << "  --  Checks: " << gparams.all_op[i]->hist_check.col(gparams.last_iiter).transpose() << endl
        // << "  --  Vals: " << temp.transpose() << endl
        << endl;
    }

    double final_GA;

    opA->Ax_temp.setZero();
    opA->forward(out, opA->Ax_temp, false, 0, true);
    final_GA = opA->Ax_temp.squaredNorm();

    cout << endl << "!!!!!! Final GA = " << final_GA << endl << endl;
     

    cout << "Done acoustic_v3!" << endl << endl;

}



void girf_ec_v1(double dt, double T, double gmax, double smax, double *G0_in, 
                complex<double> *H_in, double *girf_win_in, int N_H,
                int N_moments, double *moments_params,
                double l2_weight, double a_weight, int verbose,
                int p_iter, int p_cg_iter, int p_obj_min, double p_obj_scale,
                double **out0, double **out1, double **out2, int **outsize) 
{
    VectorXcd H_load;
    H_load.setOnes(N_H);

    VectorXd girf_win;
    girf_win.setOnes(N_H);
    
    for (int i=0; i<N_H; i++) {
        H_load(i) = H_in[i];
        girf_win(i) = girf_win_in[i];
    }

    cout << endl << "H_load:" << endl;
    cout << H_load.head(10) << endl << endl;
    
    
    int N = (int)(T/dt) + 1;
    int Naxis = 1;

    VectorXd inv_vec;
    inv_vec.setOnes(Naxis*N);

    VectorXd set_vals;
    set_vals.setOnes(Naxis*N);
    set_vals.array() *= -9999999.0;

    VectorXd fixer;
    fixer.setOnes(Naxis*N);


    for(int i = 0; i < Naxis*N; i++) {
        if (G0_in[i] > -10000) {
            set_vals(i) = G0_in[i];
            fixer(i) = 0.0;
        }
    }


    Op_Moments *opM = new Op_Moments(N, Naxis, dt, N_moments);
    MatrixXd moments;
    moments.setZero(N_moments,7);
    for (int i = 0; i < N_moments; i++) {
        for (int j = 0; j < 7; j++) {
            moments(i,j) = moments_params[j + i*7];
        }
    }
    opM->set_params(moments);

    Op_Slew *opS = new Op_Slew(N, Naxis, dt);
    Op_Gradient *opG = new Op_Gradient(N, Naxis, dt);
    Op_Duty *opD = new Op_Duty(N, Naxis, dt);
    Op_GirfEC *opGEC = new Op_GirfEC(N, Naxis, dt, H_load, girf_win, N_H);
    
    opS->set_params(smax);
    opG->set_params(gmax, set_vals);
    
    opD->weight(0) = l2_weight;
    opGEC->weight(0) = a_weight;
    // opA->set_fixer(fixer);
    // opA->set_inv_vec(inv_vec);
    
    vector<Operator*> all_op;
    all_op.push_back(opG);
    all_op.push_back(opS);
    all_op.push_back(opM);

    vector<Operator*> all_obj;
    if (l2_weight > 0) {
        all_obj.push_back(opD);
    } 
    if (a_weight > 0) {
        all_obj.push_back(opGEC);
    }

    VectorXd X;
    X.setOnes(Naxis*N);
    X.array() *= gmax/10.0;

    for(int i = 0; i < Naxis*N; i++) {
        if (set_vals(i) > -10000) {
            X(i) = set_vals(i);
        }
    }


    GroptParams gparams; 
    gparams.N = N;
    gparams.Naxis = Naxis;
    gparams.X0 = X;
    gparams.all_op = all_op; 
    gparams.all_obj = all_obj;
    gparams.inv_vec = inv_vec;
    gparams.fixer = fixer;
    gparams.set_vals = set_vals;
    gparams.verbose = verbose;


    gparams.N_iter = p_iter;
    gparams.N_feval = 500000;
    gparams.cg_resid_tol = 1.0e-3;
    gparams.d_obj_thresh = 1.0e-3;
    gparams.cg_niter = p_cg_iter;
    gparams.verbose_int = 50;

    gparams.obj_min = p_obj_min;
    gparams.obj_scale = p_obj_scale;
    
    gparams.set_vecs();
    gparams.update_vals();


    VectorXd out;
    optimize(gparams, out);

    cout << "Done!  Final n_feval = " << gparams.total_n_feval << endl;

    opM->Ax_temp.setZero();
    opM->forward(out, opM->Ax_temp, false, 0, true);

    cout << "opM =  " << opM->Ax_temp.transpose() << endl;

    
    int N_out0 = out.size();
    *out0 = new double[N_out0];
    for(int i = 0; i < N_out0; i++) {
        out0[0][i] = out(i);
    }

    int N_out1 = 16;
    *out1 = new double[N_out1];
    out1[0][0] = gparams.final_good;
    out1[0][1] = gparams.last_iiter;
    out1[0][2] = gparams.total_n_feval;

    // VectorXd outvec2;
    // outvec2.setZero(out.size());
    // opA->add2AtAx(out, outvec2);

    VectorXd outvec2;
    outvec2.setZero(opGEC->Y0.size());
    opGEC->forward(out, outvec2, false, 0, true);


    int N_out2 = outvec2.size();
    *out2 = new double[N_out2];
    for(int i = 0; i < outvec2.size(); i++) {
        out2[0][i] = outvec2(i);
    }


    *outsize = new int[3];
    outsize[0][0] = N_out0;
    outsize[0][1] = N_out1;
    outsize[0][2] = N_out2;


    for (int i = 0; i < gparams.all_op.size(); i++) {
        VectorXd temp;
        temp.setZero(all_op[i]->Y0.size());
        gparams.all_op[i]->forward(out, temp, false, 0, true);

        cout << "***** " << gparams.all_op[i]->name << " *****" << endl
        << "  --  Checks: " << gparams.all_op[i]->hist_check.col(gparams.last_iiter).transpose() << endl
        // << "  --  Vals: " << temp.transpose() << endl
        << endl;
    }

    double final_GA;

    opGEC->Ax_temp.setZero();
    opGEC->forward(out, opGEC->Ax_temp, false, 0, true);
    final_GA = opGEC->Ax_temp.squaredNorm();

    cout << endl << "!!!!!! Final GA = " << final_GA << endl << endl;
     

    cout << "Done acoustic_v3!" << endl << endl;

}




void girf_ec_v2(double dt, double T, double gmax, double smax, double *G0_in, 
                complex<double> *H_in, double *girf_win_in, int N_H,
                int N_moments, double *moments_params,
                double l2_weight, double a_weight, double stim_thresh, int verbose,
                int p_iter, int p_cg_iter, int p_obj_min, double p_obj_scale,
                double **out0, double **out1, double **out2, int **outsize) 
{
    VectorXcd H_load;
    H_load.setOnes(N_H);

    VectorXd girf_win;
    girf_win.setOnes(N_H);
    
    for (int i=0; i<N_H; i++) {
        H_load(i) = H_in[i];
        girf_win(i) = girf_win_in[i];
    }

    cout << endl << "H_load:" << endl;
    cout << H_load.head(10) << endl << endl;
    
    
    int N = (int)(T/dt) + 1;
    int Naxis = 1;

    VectorXd inv_vec;
    inv_vec.setOnes(Naxis*N);

    VectorXd set_vals;
    set_vals.setOnes(Naxis*N);
    set_vals.array() *= -9999999.0;

    VectorXd fixer;
    fixer.setOnes(Naxis*N);


    for(int i = 0; i < Naxis*N; i++) {
        if (G0_in[i] > -10000) {
            set_vals(i) = G0_in[i];
            fixer(i) = 0.0;
        }
    }


    Op_Moments *opM = new Op_Moments(N, Naxis, dt, N_moments);
    MatrixXd moments;
    moments.setZero(N_moments,7);
    for (int i = 0; i < N_moments; i++) {
        for (int j = 0; j < 7; j++) {
            moments(i,j) = moments_params[j + i*7];
        }
    }
    opM->set_params(moments);

    Op_Slew *opS = new Op_Slew(N, Naxis, dt);
    Op_Gradient *opG = new Op_Gradient(N, Naxis, dt);
    Op_Duty *opD = new Op_Duty(N, Naxis, dt);
    Op_GirfEC *opGEC = new Op_GirfEC(N, Naxis, dt, H_load, girf_win, N_H);
    
    Op_PNS *opP = new Op_PNS(N, Naxis, dt);
    opP->set_params(stim_thresh);

    opS->set_params(smax);
    opG->set_params(gmax, set_vals);
    
    opD->weight(0) = l2_weight;
    opGEC->weight(0) = a_weight;
    // opA->set_fixer(fixer);
    // opA->set_inv_vec(inv_vec);
    
    vector<Operator*> all_op;
    all_op.push_back(opG);
    all_op.push_back(opS);
    all_op.push_back(opM);

    vector<Operator*> all_obj;
    if (l2_weight > 0) {
        all_obj.push_back(opD);
    } 
    if (a_weight > 0) {
        all_obj.push_back(opGEC);
    }
    if (stim_thresh > 0) {
        all_op.push_back(opP);
    }

    VectorXd X;
    X.setOnes(Naxis*N);
    X.array() *= gmax/10.0;

    for(int i = 0; i < Naxis*N; i++) {
        if (set_vals(i) > -10000) {
            X(i) = set_vals(i);
        }
    }


    GroptParams gparams; 
    gparams.N = N;
    gparams.Naxis = Naxis;
    gparams.X0 = X;
    gparams.all_op = all_op; 
    gparams.all_obj = all_obj;
    gparams.inv_vec = inv_vec;
    gparams.fixer = fixer;
    gparams.set_vals = set_vals;
    gparams.verbose = verbose;


    gparams.N_iter = p_iter;
    gparams.N_feval = 500000;
    gparams.cg_resid_tol = 1.0e-3;
    gparams.d_obj_thresh = 1.0e-3;
    gparams.cg_niter = p_cg_iter;
    gparams.verbose_int = 50;

    gparams.obj_min = p_obj_min;
    gparams.obj_scale = p_obj_scale;
    
    gparams.set_vecs();
    gparams.update_vals();


    VectorXd out;
    optimize(gparams, out);

    cout << "Done!  Final n_feval = " << gparams.total_n_feval << endl;

    opM->Ax_temp.setZero();
    opM->forward(out, opM->Ax_temp, false, 0, true);

    cout << "opM =  " << opM->Ax_temp.transpose() << endl;

    
    int N_out0 = out.size();
    *out0 = new double[N_out0];
    for(int i = 0; i < N_out0; i++) {
        out0[0][i] = out(i);
    }

    int N_out1 = 16;
    *out1 = new double[N_out1];
    out1[0][0] = gparams.final_good;
    out1[0][1] = gparams.last_iiter;
    out1[0][2] = gparams.total_n_feval;

    // VectorXd outvec2;
    // outvec2.setZero(out.size());
    // opA->add2AtAx(out, outvec2);

    VectorXd outvec2;
    outvec2.setZero(opGEC->Y0.size());
    opGEC->forward(out, outvec2, false, 0, true);


    int N_out2 = outvec2.size();
    *out2 = new double[N_out2];
    for(int i = 0; i < outvec2.size(); i++) {
        out2[0][i] = outvec2(i);
    }


    *outsize = new int[3];
    outsize[0][0] = N_out0;
    outsize[0][1] = N_out1;
    outsize[0][2] = N_out2;


    for (int i = 0; i < gparams.all_op.size(); i++) {
        VectorXd temp;
        temp.setZero(all_op[i]->Y0.size());
        gparams.all_op[i]->forward(out, temp, false, 0, true);

        cout << "***** " << gparams.all_op[i]->name << " *****" << endl
        << "  --  Checks: " << gparams.all_op[i]->hist_check.col(gparams.last_iiter).transpose() << endl
        // << "  --  Vals: " << temp.transpose() << endl
        << endl;
    }

    double final_GA;

    opGEC->Ax_temp.setZero();
    opGEC->forward(out, opGEC->Ax_temp, false, 0, true);
    final_GA = opGEC->Ax_temp.squaredNorm();

    cout << endl << "!!!!!! Final GA = " << final_GA << endl << endl;
     

    cout << "Done girf_ec_v2!" << endl << endl;

}


void accel_test_v1(double dt, double T, double gmax, double smax, double M1, double M2,
                   double l2_weight, int do_m2, int verbose,
                   double **out0, double **out1, double **out2, int **outsize) 
{   
    
    int N = (int)(T/dt) + 1;
    int Naxis = 1;

    cout << "Debug 1" << endl << endl;

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



    int N_moments;
    if (do_m2 > 0) {
        N_moments = 3;
    } else {
        N_moments = 2;
    }

    cout << "Debug 2" << endl << endl;

    Op_Moments *opM = new Op_Moments(N, Naxis, dt, N_moments);
    MatrixXd moments(N_moments,7);

    if (do_m2 > 0) {
        moments << 0, 0, 0, 0, 0, 0.0, 1e-6,
                   0, 1, 0, 0, 0, M1, 1e-6,
                   0, 2, 0, 0, 0, M2, 1e-6;
    } else {
        moments << 0, 0, 0, 0, 0, 0.0, 1e-6,
                   0, 1, 0, 0, 0, M1, 1e-6;
    }
    opM->set_params(moments);

    cout << "Debug 3" << endl << endl;

    Op_Slew *opS = new Op_Slew(N, Naxis, dt);
    Op_Gradient *opG = new Op_Gradient(N, Naxis, dt);
    Op_Duty *opD = new Op_Duty(N, Naxis, dt);
    
    opS->set_params(smax);
    opG->set_params(gmax, set_vals);
    
    opD->weight(0) = l2_weight;
    
    vector<Operator*> all_op;
    all_op.push_back(opG);
    all_op.push_back(opS);
    all_op.push_back(opM);

    vector<Operator*> all_obj;
    if (l2_weight > 0) {
        all_obj.push_back(opD);
    }

    cout << "Debug 4" << endl << endl;

    VectorXd X;
    X.setOnes(Naxis*N);
    X.array() *= gmax/10.0;

    for(int i = 0; i < Naxis*N; i++) {
        if (set_vals(i) > -10000) {
            X(i) = set_vals(i);
        }
    }


    GroptParams gparams; 
    gparams.N = N;
    gparams.Naxis = Naxis;
    gparams.X0 = X;
    gparams.all_op = all_op; 
    gparams.all_obj = all_obj;
    gparams.inv_vec = inv_vec;
    gparams.fixer = fixer;
    gparams.set_vals = set_vals;
    gparams.verbose = verbose;

    gparams.N_iter = 500;
    gparams.N_feval = 10000;
    gparams.cg_resid_tol = 1.0e-3;
    gparams.d_obj_thresh = 1.0e-3;

    gparams.set_vecs();
    gparams.update_vals();


    VectorXd out;
    optimize(gparams, out);

    cout << "Done!  Final n_feval = " << gparams.total_n_feval << endl;

    opM->Ax_temp.setZero();
    opM->forward(out, opM->Ax_temp, false, 0, true);

    cout << "opM =  " << opM->Ax_temp.transpose() << endl;

    
    int N_out0 = out.size();
    *out0 = new double[N_out0];
    for(int i = 0; i < N_out0; i++) {
        out0[0][i] = out(i);
    }

    int N_out1 = 16;
    *out1 = new double[N_out1];
    out1[0][0] = gparams.final_good;
    out1[0][1] = gparams.last_iiter;
    out1[0][2] = gparams.total_n_feval;


    *outsize = new int[3];
    outsize[0][0] = N_out0;
    outsize[0][1] = N_out1;


    for (int i = 0; i < gparams.all_op.size(); i++) {
        VectorXd temp;
        temp.setZero(all_op[i]->Y0.size());
        gparams.all_op[i]->forward(out, temp, false, 0, true);

        cout << "***** " << gparams.all_op[i]->name << " *****" << endl
        << "  --  Checks: " << gparams.all_op[i]->hist_check.col(gparams.last_iiter).transpose() << endl
        // << "  --  Vals: " << temp.transpose() << endl
        << endl;
    }     

    cout << "Done accel_test_v1!" << endl << endl;

}



void simple_bipolar_pns(double dt, double T, double gmax, double smax, double M0, double M1,
                        double stim_thresh, double l2_weight, int verbose,
                        double **out0, double **out1, double **out2, int **outsize) 
{   
    
    int N = (int)(T/dt) + 1;
    int Naxis = 1;

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



    int N_moments = 2;
    Op_Moments *opM = new Op_Moments(N, Naxis, dt, N_moments);
    MatrixXd moments(N_moments,7);

    moments << 0, 0, 0, 0, 0, M0, 1e-6,
                0, 1, 0, 0, 0, M1, 1e-6;

    opM->set_params(moments);


    Op_PNS *opP = new Op_PNS(N, Naxis, dt);
    Op_Slew *opS = new Op_Slew(N, Naxis, dt);
    Op_Gradient *opG = new Op_Gradient(N, Naxis, dt);
    Op_Duty *opD = new Op_Duty(N, Naxis, dt);
    
    opS->set_params(smax);
    opG->set_params(gmax, set_vals);
    opP->set_params(stim_thresh);
    
    opD->weight(0) = l2_weight;
    
    vector<Operator*> all_op;
    all_op.push_back(opG);
    all_op.push_back(opS);
    all_op.push_back(opM);
    if (stim_thresh > 0) {
        all_op.push_back(opP);
    }

    vector<Operator*> all_obj;
    if (l2_weight > 0) {
        all_obj.push_back(opD);
    }


    VectorXd X;
    X.setOnes(Naxis*N);
    X.array() *= gmax/10.0;

    for(int i = 0; i < Naxis*N; i++) {
        if (set_vals(i) > -10000) {
            X(i) = set_vals(i);
        }
    }


    GroptParams gparams; 
    gparams.N = N;
    gparams.Naxis = Naxis;
    gparams.X0 = X;
    gparams.all_op = all_op; 
    gparams.all_obj = all_obj;
    gparams.inv_vec = inv_vec;
    gparams.fixer = fixer;
    gparams.set_vals = set_vals;
    gparams.verbose = verbose;

    gparams.N_iter = 500;
    gparams.N_feval = 10000;
    gparams.cg_resid_tol = 1.0e-3;
    gparams.d_obj_thresh = 1.0e-3;

    gparams.set_vecs();
    gparams.update_vals();


    VectorXd out;
    optimize(gparams, out);

    if (verbose > 0) {
        cout << "Done!  Final n_feval = " << gparams.total_n_feval << endl;
    }

    opM->Ax_temp.setZero();
    opM->forward(out, opM->Ax_temp, false, 0, true);

    if (verbose > 0) {
        cout << "opM =  " << opM->Ax_temp.transpose() << endl;
    }
    
    int N_out0 = out.size();
    *out0 = new double[N_out0];
    for(int i = 0; i < N_out0; i++) {
        out0[0][i] = out(i);
    }

    int N_out1 = 16;
    *out1 = new double[N_out1];
    out1[0][0] = gparams.final_good;
    out1[0][1] = gparams.last_iiter;
    out1[0][2] = gparams.total_n_feval;


    int N_out2 = 16;
    *out2 = new double[N_out2];
    out2[0][0] = gparams.final_good;
    out2[0][1] = gparams.last_iiter;
    out2[0][2] = gparams.total_n_feval;


    *outsize = new int[3];
    outsize[0][0] = N_out0;
    outsize[0][1] = N_out1;
    outsize[0][2] = N_out2;

}





void cones_pns_3(double dt, int N, double gmax, double smax, double *G0_in,
                int N_moments, double *moments_in,
                double eddy_lam, int eddy_stop, double eddy_tol,
               double stim_thresh,  double l2_weight, int rot_var_mode, int verbose,
               int p_iter, int p_cg_iter, int p_obj_min, double p_obj_scale,
               double **out0, double **out1, double **out2, int **outsize) 
{
    int Naxis = 3;

    VectorXd inv_vec;
    inv_vec.setOnes(Naxis*N);

    VectorXd set_vals;
    set_vals.setOnes(Naxis*N);
    set_vals.array() *= -9999999.0;

    VectorXd fixer;
    fixer.setOnes(Naxis*N);

    for(int i = 0; i < Naxis*N; i++) {
        if (G0_in[i] > -10000) {
            set_vals(i) = G0_in[i];
            fixer(i) = 0.0;
        }
    }


    Op_Moments *opM = new Op_Moments(N, Naxis, dt, N_moments);
    MatrixXd moments(N_moments,7);
    for (int j = 0; j < N_moments; j++) {
        for (int i = 0; i < 7; i++) {
            moments(j, i) = moments_in[j*7 + i];
        }
    }
    opM->set_params(moments);

    Op_PNS *opP = new Op_PNS(N, Naxis, dt);
    Op_Slew *opS = new Op_Slew(N, Naxis, dt);
    Op_Gradient *opG = new Op_Gradient(N, Naxis, dt);
    Op_Duty *opD = new Op_Duty(N, Naxis, dt);
    
    opS->set_params(smax);
    opG->set_params(gmax, set_vals);
    opP->set_params(stim_thresh);

    Op_Eddy *opE = new Op_Eddy(N, Naxis, dt, 1);
    opE->prep_A(eddy_lam, eddy_stop, eddy_tol);

    if (rot_var_mode == 0) {
        opG->rot_variant = false;
        opS->rot_variant = false;
    } else {
        opG->rot_variant = true;
        opS->rot_variant = true;
    }
    
    opD->weight(0) = l2_weight;
    
    vector<Operator*> all_op;
    all_op.push_back(opG);
    all_op.push_back(opS);
    all_op.push_back(opM);
    if (stim_thresh > 0) {
        all_op.push_back(opP);
    }
    if (eddy_tol > 0) {
        all_op.push_back(opE);
    }

    vector<Operator*> all_obj;
    if (l2_weight > 0) {
        all_obj.push_back(opD);
    }



    VectorXd X;
    X.setOnes(Naxis*N);
    X.array() *= gmax/10.0;

    for(int i = 0; i < Naxis*N; i++) {
        if (set_vals(i) > -10000) {
            X(i) = set_vals(i);
        }
    }


    GroptParams gparams; 
    gparams.N = N;
    gparams.Naxis = Naxis;
    gparams.X0 = X;
    gparams.all_op = all_op; 
    gparams.all_obj = all_obj;
    gparams.inv_vec = inv_vec;
    gparams.fixer = fixer;
    gparams.set_vals = set_vals;

    // gparams.N_iter = 2000;
    // gparams.N_feval = 50000;
    // gparams.cg_resid_tol = 1.0e-3;
    // gparams.d_obj_thresh = 1.0e-3;

    gparams.verbose = verbose;
    gparams.N_iter = p_iter;
    gparams.N_feval = 500000;
    gparams.cg_resid_tol = 1.0e-3;
    gparams.d_obj_thresh = 1.0e-3;
    gparams.cg_niter = p_cg_iter;
    gparams.verbose_int = 50;

    gparams.obj_min = p_obj_min;
    gparams.obj_scale = p_obj_scale;
    

    gparams.set_vecs();
    gparams.update_vals();


    VectorXd out;
    optimize(gparams, out);

    cout << "Done!  Final n_feval = " << gparams.total_n_feval << endl;

    opM->Ax_temp.setZero();
    opM->forward(out, opM->Ax_temp, false, 0, true);
    cout << "opM =  " << opM->Ax_temp.transpose() << endl;

    opE->Ax_temp.setZero();
    opE->forward(out, opE->Ax_temp, false, 0, true);
    cout << "opE =  " << opE->Ax_temp.transpose() << endl;

    
    int N_out0 = out.size();
    *out0 = new double[N_out0];
    for(int i = 0; i < N_out0; i++) {
        out0[0][i] = out(i);
    }

    int N_out1 = 16;
    *out1 = new double[N_out1];
    out1[0][0] = gparams.final_good;
    out1[0][1] = gparams.last_iiter;
    out1[0][2] = gparams.total_n_feval;

    // int N_out1 = 16;
    // *out1 = new double[N_out1];
    // out1[0][0] = gparams.total_n_feval;
    // out1[0][1] = gparams.all_obj[0]->current_obj;
    // out1[0][2] = time_taken1 * 1e-6;
    // out1[0][3] = gparams.last_iiter;

    // int N_out2 = 160;
    // VectorXd v_out2;
    // v_out2.setOnes(N_out2);

    // interp_vec2vec(out, v_out2);

    // *out2 = new double[N_out2];
    // for(int i = 0; i < N_out2; i++) {
    //     out2[0][i] = v_out2(i);
    // }


    *outsize = new int[3];
    outsize[0][0] = N_out0;
    outsize[0][1] = N_out1;
    // outsize[0][2] = N_out2;


    for (int i = 0; i < gparams.all_op.size(); i++) {
        VectorXd temp;
        temp.setZero(all_op[i]->Y0.size());
        gparams.all_op[i]->forward(out, temp, false, 0, true);

        cout << "***** " << gparams.all_op[i]->name << " *****" << endl
        << "  --  Checks: " << gparams.all_op[i]->hist_check.col(gparams.last_iiter).transpose() << endl
        // << "  --  Vals: " << temp.transpose() << endl
        << endl;
    }
    

    cout << "Done cones_pns_3!" << endl << endl;

}



void cones_pns_3_v2(double dt, int N, double gmax, double smax, double *G0_in,
                int N_moments, double *moments_in,
                double *eddy_lam, int eddy_stop, double eddy_tol, int Nlam,
               double stim_thresh,  double l2_weight, int rot_var_mode, int verbose,
               int p_iter, int p_cg_iter, int p_obj_min, double p_obj_scale,
               double **out0, double **out1, double **out2, int **outsize) 
{
    int Naxis = 3;

    VectorXd lam_in;
    lam_in.setZero(Nlam);
    
    for (int i=0; i<Nlam; i++) {
        lam_in(i) = eddy_lam[i];
    }

    VectorXd inv_vec;
    inv_vec.setOnes(Naxis*N);

    VectorXd set_vals;
    set_vals.setOnes(Naxis*N);
    set_vals.array() *= -9999999.0;

    VectorXd fixer;
    fixer.setOnes(Naxis*N);

    for(int i = 0; i < Naxis*N; i++) {
        if (G0_in[i] > -10000) {
            set_vals(i) = G0_in[i];
            fixer(i) = 0.0;
        }
    }


    Op_Moments *opM = new Op_Moments(N, Naxis, dt, N_moments);
    MatrixXd moments(N_moments,7);
    for (int j = 0; j < N_moments; j++) {
        for (int i = 0; i < 7; i++) {
            moments(j, i) = moments_in[j*7 + i];
        }
    }
    opM->set_params(moments);

    Op_PNS *opP = new Op_PNS(N, Naxis, dt);
    Op_Slew *opS = new Op_Slew(N, Naxis, dt);
    Op_Gradient *opG = new Op_Gradient(N, Naxis, dt);
    Op_Duty *opD = new Op_Duty(N, Naxis, dt);
    
    opS->set_params(smax);
    opG->set_params(gmax, set_vals);
    opP->set_params(stim_thresh);

    Op_Eddy *opE = new Op_Eddy(N, Naxis, dt, Nlam);
    opE->prep_A(lam_in, eddy_stop, eddy_tol);

    if (rot_var_mode == 0) {
        opG->rot_variant = false;
        opS->rot_variant = false;
    } else {
        opG->rot_variant = true;
        opS->rot_variant = true;
    }
    
    opD->weight(0) = l2_weight;
    
    vector<Operator*> all_op;
    all_op.push_back(opG);
    all_op.push_back(opS);
    all_op.push_back(opM);
    if (stim_thresh > 0) {
        all_op.push_back(opP);
    }
    if (eddy_tol > 0) {
        all_op.push_back(opE);
    }

    vector<Operator*> all_obj;
    if (l2_weight > 0) {
        all_obj.push_back(opD);
    }



    VectorXd X;
    X.setOnes(Naxis*N);
    X.array() *= gmax/10.0;

    for(int i = 0; i < Naxis*N; i++) {
        if (set_vals(i) > -10000) {
            X(i) = set_vals(i);
        }
    }


    GroptParams gparams; 
    gparams.N = N;
    gparams.Naxis = Naxis;
    gparams.X0 = X;
    gparams.all_op = all_op; 
    gparams.all_obj = all_obj;
    gparams.inv_vec = inv_vec;
    gparams.fixer = fixer;
    gparams.set_vals = set_vals;

    // gparams.N_iter = 2000;
    // gparams.N_feval = 50000;
    // gparams.cg_resid_tol = 1.0e-3;
    // gparams.d_obj_thresh = 1.0e-3;

    gparams.verbose = verbose;
    gparams.N_iter = p_iter;
    gparams.N_feval = 500000;
    gparams.cg_resid_tol = 1.0e-3;
    gparams.d_obj_thresh = 1.0e-3;
    gparams.cg_niter = p_cg_iter;
    gparams.verbose_int = 50;

    gparams.obj_min = p_obj_min;
    gparams.obj_scale = p_obj_scale;
    

    gparams.set_vecs();
    gparams.update_vals();


    VectorXd out;
    optimize(gparams, out);

    cout << "Done!  Final n_feval = " << gparams.total_n_feval << endl;

    opM->Ax_temp.setZero();
    opM->forward(out, opM->Ax_temp, false, 0, true);
    cout << "opM =  " << opM->Ax_temp.transpose() << endl;

    opE->Ax_temp.setZero();
    opE->forward(out, opE->Ax_temp, false, 0, true);
    cout << "opE =  " << opE->Ax_temp.transpose() << endl;

    
    int N_out0 = out.size();
    *out0 = new double[N_out0];
    for(int i = 0; i < N_out0; i++) {
        out0[0][i] = out(i);
    }

    int N_out1 = 16;
    *out1 = new double[N_out1];
    out1[0][0] = gparams.final_good;
    out1[0][1] = gparams.last_iiter;
    out1[0][2] = gparams.total_n_feval;

    // int N_out1 = 16;
    // *out1 = new double[N_out1];
    // out1[0][0] = gparams.total_n_feval;
    // out1[0][1] = gparams.all_obj[0]->current_obj;
    // out1[0][2] = time_taken1 * 1e-6;
    // out1[0][3] = gparams.last_iiter;

    // int N_out2 = 160;
    // VectorXd v_out2;
    // v_out2.setOnes(N_out2);

    // interp_vec2vec(out, v_out2);

    // *out2 = new double[N_out2];
    // for(int i = 0; i < N_out2; i++) {
    //     out2[0][i] = v_out2(i);
    // }


    *outsize = new int[3];
    outsize[0][0] = N_out0;
    outsize[0][1] = N_out1;
    // outsize[0][2] = N_out2;


    for (int i = 0; i < gparams.all_op.size(); i++) {
        VectorXd temp;
        temp.setZero(all_op[i]->Y0.size());
        gparams.all_op[i]->forward(out, temp, false, 0, true);

        cout << "***** " << gparams.all_op[i]->name << " *****" << endl
        << "  --  Checks: " << gparams.all_op[i]->hist_check.col(gparams.last_iiter).transpose() << endl
        // << "  --  Vals: " << temp.transpose() << endl
        << endl;
    }
    

    cout << "Done cones_pns_3!" << endl << endl;

}


#include "pocketfft_hdronly.h"

void girf_tester_v1(int N, double *G, complex<double> *H_in, 
                complex<double> **out0, complex<double> **out1, complex<double> **out2, int **outsize)
{

    cout << "Starting girf_tester_v1" << endl << endl;

    pocketfft::shape_t shape;
    pocketfft::stride_t stride_d;
    pocketfft::stride_t stride_cd;
    pocketfft::shape_t axes;

    shape.push_back(N);
    stride_d.push_back(sizeof(complex<double>));
    stride_cd.push_back(sizeof(complex<double>));
    axes.push_back(0);

    VectorXcd H = Map<VectorXcd>(H_in, N, 1);

    *out0 = new complex<double>[N];

    pocketfft::c2c(shape, stride_d, stride_cd, axes, pocketfft::FORWARD,
                    H.data(), *out0, 1.);

    *outsize = new int[3];
    outsize[0][0] = N;

    cout << "Done with girf_tester_v1" << endl << endl;

}
