#include <iostream>
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
#include "cg_iter.h"
#include "gropt_params.h"
#include "optimize.h"
#include "diff_utils.h"

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

    Op_Moments *opM = new Op_Moments(N, dt, N_moments);
    Op_Slew *opS = new Op_Slew(N, dt);
    Op_Gradient *opG = new Op_Gradient(N, dt);
    Op_BVal *opB = new Op_BVal(N, dt);

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