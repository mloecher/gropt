#include "gropt_params.h"
#include "optimize.h"

GroptParams::GroptParams() {
    Naxis = 1;
    N_iter = 5000;
    N_feval = 30000;

    cg_niter = 10000;
    cg_resid_tol = 5.0e-3;
    cg_abs_tol = 1.0e-16;

    obj_min = 0;
    obj_scale = 1.0;

    grw_interval = 16;
    grw_start = 32;
    grw_scale = 8.0;

    cushion = 1e-2;

    // reweighting defaults here
    rw_scalelim = 1.5;
    rw_interval = 16;
    rw_eps = 1.0e-16;
    e_corr = 0.5;
    weight_min = 1.0e-4;
    weight_max = 1.0e64;

    d_obj_thresh = 1e-4;

    do_init = true;
    verbose = 0;
    verbose_int = 20;

    final_good = 0;
}

GroptParams::~GroptParams() {
    for (int i = 0; i < all_op.size(); i++) {
        delete all_op[i];
    }

    for (int i = 0; i < all_obj.size(); i++) {
        delete all_obj[i];
    }
}

// Use this function to propogate the values to the individual constraints
// Mainly needed to reset other values when cushion changes
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

void GroptParams::set_vecs()
{
    for (int i = 0; i < all_op.size(); i++) {
        all_op[i]->set_inv_vec(inv_vec);
        all_op[i]->set_fixer(fixer);
    }

    for (int i = 0; i < all_obj.size(); i++) {
        all_obj[i]->set_inv_vec(inv_vec);
        all_obj[i]->set_fixer(fixer);
    }
}

void GroptParams::defaults_diffusion()
{
    rw_scalelim = 1.5;
    rw_interval = 16;
    rw_eps = 1.0e-16;

    grw_interval = 16;
    grw_start = 32;
    grw_scale = 8.0;

    cg_niter = 50;
    cg_resid_tol = 1.0e-2;
    cg_abs_tol = 1.0e-16;

    e_corr = 0.6;
    weight_min = 1.0e-4;
    weight_max = 1.0e64;
    d_obj_thresh = 1e-4;

    update_vals();
}


void GroptParams::interp_from_gparams(GroptParams &gparams_in, VectorXd &X_in)
{
    double mod = 0.5;
    interp_vec2vec(X_in, X0);
    for (int i = 0; i < all_op.size(); i++) {
        all_op[i]->weight = gparams_in.all_op[i]->weight; 
        // all_op[i]->weight.setOnes();
        // all_op[i]->gamma = gparams_in.all_op[i]->gamma; 
        all_op[i]->gamma.setOnes();
        interp_vec2vec(gparams_in.all_op[i]->U0, all_op[i]->U0);
        interp_vec2vec(gparams_in.all_op[i]->Y0, all_op[i]->Y0);
        cout << "interp_from_gparams " << i << "  " << gparams_in.all_op[i]->U0.squaredNorm() << "  " << all_op[i]->U0.squaredNorm() << endl;
        if (all_op[i]->do_rw) {
            all_op[i]->weight *= mod;
            gparams_in.all_op[i]->U0 *= mod;
        }
    }

    for (int i = 0; i < all_obj.size(); i++) {        
        all_obj[i]->weight = gparams_in.all_obj[i]->weight;  
        // all_obj[i]->gamma = gparams_in.all_obj[i]->gamma; 
        // all_obj[i]->weight.setOnes();
        all_obj[i]->gamma.setOnes();
        interp_vec2vec(gparams_in.all_obj[i]->U0, all_obj[i]->U0);
        interp_vec2vec(gparams_in.all_obj[i]->Y0, all_obj[i]->Y0);
        cout << "interp_from_gparams " << i << "  " << all_obj[i]->U0.squaredNorm() << endl;
    }

}