#ifndef WRAPPERS_H
#define WRAPPERS_H

#include <iostream>
#ifdef USE_CHRONO
    #include <chrono>
#endif 
#include <vector>
#include <Eigen/Dense>

#include "gropt_params.h"

using namespace Eigen;

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
#endif