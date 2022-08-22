#ifndef WRAPPERS_H
#define WRAPPERS_H

#include <iostream>
#ifdef USE_CHRONO
    #include <chrono>
#endif 
#include <vector>
#include "Eigen/Dense"

#include "gropt_params.h"

using namespace Eigen;

void threed_diff(double *params0, double *params1, double **out0, double **out1, double **out2, int **outsize);

void python_wrapper_warmstart_v1(double *params0, double *params1, 
             double **out0, double **out1, double **out2, int **outsize);

void gropt_legacy(double **G_out, int *N_out, double **ddebug, int verbose, 
                  double dt0, double gmax, double smax, double TE, 
                  int N_moments, double *moments_params, double PNS_thresh, 
                  double T_readout, double T_90, double T_180, int diffmode, double dt_out,
                  int N_eddy, double *eddy_params, double search_bval, double slew_reg, int Naxis);


void gropt_diff_seq(double **G_out, int *N_out, int verbose,
                    double dt0, double dt_out, double gmax, double smax, double TE,
                    double T_readout, double T_90, double T_180, int MMT);

void python_wrapper_v1(double *params0, double *params1, 
            double **out0, double **out1, double **out2, int **outsize);

void diff_duty_cycle(double dt, double T_90, double T_180, double T_readout, double TE, 
                     int N_moments, double gmax, double smax, double bval, double duty_cycle,
                     double **out0, double **out1, double **out2, int **outsize) ;

void spect_phase_contrast(double dt, double T, double gmax, double smax, double M0, double M1, int Neddy,
                          int Nset, double *in_setvals, double es_weight, int verbose, int start_ind1,
                          double **out0, double **out1, double **out2, int **outsize) ;


void spect_phase_contrast_spoiler(double dt, double T, double gmax, double smax, int Neddy,
                            double M0_spoil, double M0_ss, double M1_ss,
                            int ind_spoil_stop, int ind_ss_start0, int ind_ss_start1,
                            int eddy_start, int eddy_end,
                          int Nset, double *in_setvals, double es_weight, int verbose, 
                          double **out0, double **out1, double **out2, int **outsize) ;

void simple_bipolar(double dt, double T, double gmax, double smax, double M0, double M1,
                   double l2_weight, int verbose,
                   double **out0, double **out1, double **out2, int **outsize); 

void spect_phase_contrast_spoiler2(double dt, double T, double gmax, double smax, int Neddy,
                            double M0_spoil, double M0_ss, double M1_ss,
                            int ind_spoil_start, int ind_spoil_stop, int ind_ss_start0, int ind_ss_start1,
                            int eddy_start, int eddy_end,
                          int Nset, double *in_setvals, double es_weight, double l2_weight, double *in_spect, int cg_iter, int verbose, 
                          double **out0, double **out1, double **out2, int **outsize) ;
                          

void en_code(double dt, double T_90, double T_180, double T_readout, double TE, 
                     int N_moments, double gmax, double smax, double lambda,
                     double **out0, double **out1, double **out2, int **outsize);

void rewinder3(double dt, double T, double gmax, double smax, double *G0_in, double *M0_in,
               double l2_weight,
               double **out0, double **out1, double **out2, int **outsize);

void acoustic_v1(double dt, double T, double gmax, double smax, double *G0_in, double *H_in,
                double l2_weight, double a_weight, int verbose,
                double **out0, double **out1, double **out2, int **outsize);

void acoustic_v2(double dt, double T, double gmax, double smax, double *G0_in, complex<double> *H_in,
                double l2_weight, double a_weight, int verbose,
                double **out0, double **out1, double **out2, int **outsize);

void acoustic_v3(double dt, double T, double gmax, double smax, double *G0_in, 
                complex<double> *H_in, int N_H,
                int N_moments, double *moments_params,
                double l2_weight, double a_weight, int verbose,
                int p_iter, int p_cg_iter, int p_obj_min, double p_obj_scale,
                double **out0, double **out1, double **out2, int **outsize) ;

void accel_test_v1(double dt, double T, double gmax, double smax, double M1, double M2,
                   double l2_weight, int do_m2, int verbose,
                   double **out0, double **out1, double **out2, int **outsize);

void girf_tester_v1(int N, double *G, complex<double> *H_in, 
                complex<double> **out0, complex<double> **out1, complex<double> **out2, int **outsize);

void girf_ec_v1(double dt, double T, double gmax, double smax, double *G0_in, 
                complex<double> *H_in, double *girf_win_in, int N_H,
                int N_moments, double *moments_params,
                double l2_weight, double a_weight, int verbose,
                int p_iter, int p_cg_iter, int p_obj_min, double p_obj_scale,
                double **out0, double **out1, double **out2, int **outsize);


void girf_ec_v2(double dt, double T, double gmax, double smax, double *G0_in, 
                complex<double> *H_in, double *girf_win_in, int N_H,
                int N_moments, double *moments_params,
                double l2_weight, double a_weight, double stim_thresh, int verbose,
                int p_iter, int p_cg_iter, int p_obj_min, double p_obj_scale,
                double **out0, double **out1, double **out2, int **outsize);


void simple_bipolar_pns(double dt, double T, double gmax, double smax, double M0, double M1,
                   double stim_thresh, double l2_weight, int verbose,
                   double **out0, double **out1, double **out2, int **outsize); 

void cones_pns_3(double dt, int N, double gmax, double smax, double *G0_in,
                int N_moments, double *moments_in,
                double eddy_lam, int eddy_stop, double eddy_tol,
               double stim_thresh,  double l2_weight, int rot_var_mode, int verbose,
               int p_iter, int p_cg_iter, int p_obj_min, double p_obj_scale,
               double **out0, double **out1, double **out2, int **outsize);

void cones_pns_3_v2(double dt, int N, double gmax, double smax, double *G0_in,
                int N_moments, double *moments_in,
                double *eddy_lam, int eddy_stop, double eddy_tol, int Nlam,
               double stim_thresh,  double l2_weight, int rot_var_mode, int verbose,
               int p_iter, int p_cg_iter, int p_obj_min, double p_obj_scale,
               double **out0, double **out1, double **out2, int **outsize);

void diff_pre_eddy(double dt, double T_90, double T_180, double T_readout, double T_pre, double TE, 
                   int moment_order, double gmax, double smax, double *eddy_lam_in, int Nlam, double maxwell_tol, double b_weight,
                   double **out0, double **out1, double **out2, int **outsize);

#endif