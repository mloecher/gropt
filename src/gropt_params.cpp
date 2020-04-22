#include "gropt_params.h"

GroptParams::GroptParams() {
    N_iter = 5000;
    N_feval = 20000;

    cg_niter = 5000;
    cg_resid_tol = 1.0e-3;
    cg_abs_tol = 1.0e-16;

    grw_interval = 100;
    grw_start = 200;
    grw_scale = 1.2;

    cushion = 1e-2;

    // reweighting defaults here
    rw_scalelim = 2.0;
    rw_interval = 8;
    rw_eps = 1.0e-6;
    e_corr = 0.5;
    weight_min = 1.0e-4;
    weight_max = 1.0e64;

    d_obj_thresh = 1e-4;
}

void GroptParams::update_vals()
{
    for (int i = 0; i < all_op.size(); i++) {
        all_op[i]->change_cushion(cushion);
        all_op[i]->rw_scalelim = rw_scalelim;
        all_op[i]->rw_interval = rw_interval;
        all_op[i]->rw_eps = rw_eps;
        all_op[i]->e_corr = e_corr;
        all_op[i]->weight_min = weight_min;
        all_op[i]->weight_max = weight_max;
    }
}

void GroptParams::defaults_diffusion()
{
    rw_scalelim = 1.5;
    rw_interval = 16;
    rw_eps = 1.0e-16;

    grw_interval = 8;
    grw_start = 400;
    grw_scale = 2.0;

    cg_niter = 10000;
    cg_resid_tol = 1.0e-3;
    cg_abs_tol = 1.0e-16;

    e_corr = 0.5;
    weight_min = 1.0e-4;
    weight_max = 1.0e64;
    d_obj_thresh = 1e-4;

    update_vals();
}