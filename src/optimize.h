#ifndef OPTIMIZE_H
#define OPTIMIZE_H

#include <iostream>
#include <vector>
#include <Eigen/Dense>

#include "gropt_params.h"

using namespace Eigen;

void optimize(GroptParams &gparams, VectorXd &out);
void interp_vec2vec(VectorXd &vec0, VectorXd &vec1); 

#endif