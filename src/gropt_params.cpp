#include "gropt_params.h"

GroptParams::GroptParams() {
  N_iter = 2000;
  N_feval = 20000;
  
  cg_niter = 5000;
  cg_resid_tol = 1.0e-2;
  cg_abs_tol = 1.0e-16;
}