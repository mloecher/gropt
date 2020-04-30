#ifndef DIFF_UTILS_H
#define DIFF_UTILS_H

#include <iostream>
#include <vector>
#include <Eigen/Dense>

#include "gropt_params.h"

using namespace Eigen;

void simple_diff(GroptParams &gparams, double dt,
                double T_90, double T_180, double T_readout, double TE,
                double gmax, double smax, int N_moments, double moment_tol, bool siemens_diff);

#endif