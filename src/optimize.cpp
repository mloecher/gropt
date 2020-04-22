#include "optimize.h"
#include "cg_iter.h"

using namespace Eigen;
using namespace std;

#define N_HIST_TEMP 20

void do_globalreweight_simple(GroptParams &gparams, int iiter)
{
    int grw_interval = 10;
    
    // First do a simple reweight based on if te check was passed
    if ((iiter > grw_interval) && (iiter%grw_interval==0)) {
        
        cout << "Global Reweight iiter = " << iiter << endl;
        for (int i = 0; i < gparams.all_op.size(); i++) {
            cout << "   Name: " << gparams.all_op[i]->name << "   " << gparams.all_op[i]->do_rw << " *** ";
            for (int j = 0; j < gparams.all_op[i]->hist_check.col(iiter).size(); j++) {
                cout << "   " << gparams.all_op[i]->hist_check(j,iiter);
                if ((gparams.all_op[i]->do_rw) && (gparams.all_op[i]->hist_check(j,iiter) > 0)) {
                    cout << "rw";
                    gparams.all_op[i]->weight(j) *= 1.1;
                }
            }
            cout << endl;
        }

    }
}

void do_globalreweight(GroptParams &gparams, int iiter)
{
    int grw_interval = gparams.grw_interval;
    int grw_start = gparams.grw_start;

    int max_i = -1;
    int max_j = -1;
    double max_feas = -1.0;
    double feas;
    
    // Only reweight worst r_feas
    if ((iiter > grw_start) && (iiter%grw_interval==0)) {
        
        // cout << "Global Reweight iiter = " << iiter << endl;
        for (int i = 0; i < gparams.all_op.size(); i++) {
            // cout << "   Name: " << gparams.all_op[i]->name << "   " << gparams.all_op[i]->do_rw << "  ";
            for (int j = 0; j < gparams.all_op[i]->hist_check.col(iiter).size(); j++) {
                // cout << "  --  " << gparams.all_op[i]->hist_check(j,iiter) << "  " << gparams.all_op[i]->hist_feas(j,iiter);
                if ((gparams.all_op[i]->do_rw) && (gparams.all_op[i]->hist_check(j,iiter) > 0)) {
                    feas = gparams.all_op[i]->hist_feas(j,iiter);
                    if (feas > max_feas) {
                        max_feas = feas;
                        max_i = i;
                        max_j = j;
                    }
                }
            }
            // cout << endl;
        }

        if (max_i >= 0) {
            // cout << "Reweighting  " << max_i << " " << max_j << " " << max_feas << endl;
            gparams.all_op[max_i]->weight(max_j) *= gparams.grw_scale;
        }

    }
}

int do_checks(GroptParams &gparams, int iiter, VectorXd &X)
{    
    // Get the current objective value
    double current_obj = gparams.all_obj[0]->hist_obj(0,iiter);
    double max_diff = 0.0;
    
    // Find the biggest objective value difference from current, in the last N_HIST_TEMP iterations
    for (int i = 0; i < N_HIST_TEMP; i++) {
        if ((iiter - i) < 0) {break;}
        double obj = gparams.all_obj[0]->hist_obj(0,iiter - i);
        double diff = fabs(current_obj - obj);
        if (diff > max_diff) {
            max_diff = diff;
        }
    }
    // Normalize
    max_diff /= current_obj;

    // Sum up all of the check values, (0 is pass so anything greater than 0 does not pass)
    double all_check = 0.0;
    for (int i = 0; i < gparams.all_op.size(); i++) {
        all_check += gparams.all_op[i]->hist_check.col(iiter).sum();
    }

    if ((max_diff < gparams.d_obj_thresh) && (all_check < 0.5)) {
        cout << "Done do_check Iter = " << iiter << "  " << current_obj << "  " << max_diff << "  " << all_check << endl;
        return 1;
    } else {
        return 0;
    }
}

void optimize(GroptParams &gparams, VectorXd &out)
{
    // Initialize starting vector
    VectorXd X;
    X = gparams.X0;

    // Initialize the CG iteration class
    CG_Iter cg(gparams.N, gparams.cg_niter, 
                gparams.cg_resid_tol, gparams.cg_abs_tol);

    // Compute Y0 = Ax for all operators
    for (int i = 0; i < gparams.all_op.size(); i++) {
        gparams.all_op[i]->init(X);
    }

    // Actual iterations
    for (int iiter = 0; iiter < gparams.N_iter; iiter++) {
        // Store what iteration we are on for later array indexing
        gparams.last_iiter = iiter;

        // Do CG iterations
        X = cg.do_CG(gparams.all_op, gparams.all_obj, X);

        // Update all constraints        
        for (int i = 0; i < gparams.all_op.size(); i++) {
            gparams.all_op[i]->update(X, iiter);
        }
        
        // Compute objective values if there are some
        for (int i = 0; i < gparams.all_obj.size(); i++) {
            gparams.all_obj[i]->get_obj(X, iiter);
        }

        // Stopping checks
        if (do_checks(gparams, iiter, X) > 0) {
            break;
        }

        // Reweight constraints
        do_globalreweight(gparams, iiter);
    }

    int n_feval = 0;
    for (int i = 0; i < cg.hist_n_iter.size(); i++) {
        n_feval += cg.hist_n_iter[i];
    }

    cout << "Final n_feval = " << n_feval << endl;

    out = X;
    gparams.total_n_feval = n_feval;

    return;
}
