#ifndef OPTIMIZE_H
#define OPTIMIZE_H

#include <iostream>
#include <chrono> 
#include <vector>
#include <Eigen/Dense>

#include "gropt_params.h"

using namespace Eigen;

void optimize(GroptParams &gparams, VectorXd &out);

#endif