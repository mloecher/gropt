#ifndef OPTIMIZE_H
#define OPTIMIZE_H

#include <iostream>
#include <chrono> 
#include <vector>
#include <Eigen/Dense>

#include "gropt_params.h"

using namespace Eigen;

class OptHist
{
    public:
        VectorXd check_hist;
        VectorXd obj_hist;

        OptHist(GroptParams &gparams);
};

void optimize(GroptParams &gparams, VectorXd &out);

#endif