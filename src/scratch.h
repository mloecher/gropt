#ifndef SCRATCH_H
#define SCRATCH_H

#include <iostream>
#ifdef USE_CHRONO
    #include <chrono>
#endif 
#include <vector>
#include <Eigen/Dense>

#include "gropt_params.h"

using namespace Eigen;

void oned_flowcomp(); 
void threed_flowcomp(); 

#endif