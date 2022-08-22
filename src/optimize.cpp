#include "optimize.h"
#include "cg_iter.h"

using namespace Eigen;
using namespace std;

#define N_HIST_TEMP 20

// This is an old and simpler reweighting function that just ups the weight on every constraint that hasn't passed its check
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

// Reweight the 'worst' constraint, based on the r_feas value from the PAR-SDMM paper
void do_globalreweight(GroptParams &gparams, int iiter)
{
    int grw_interval = gparams.grw_interval;
    int grw_start = gparams.grw_start;

    int max_i = -1;
    int max_j = -1;
    double max_feas = -1.0;
    double feas;
    
    // Find the worst r_feas (search each constraint, and sub-constraints if applicable (such as in moments))
    if ((iiter > grw_start) && (iiter%grw_interval==0)) {
        
        for (int i = 0; i < gparams.all_op.size(); i++) {
            for (int j = 0; j < gparams.all_op[i]->hist_check.col(iiter).size(); j++) {
                if ((gparams.all_op[i]->do_rw) && (gparams.all_op[i]->hist_check(j,iiter) > 0)) {
                    feas = gparams.all_op[i]->hist_feas(j,iiter);
                    if (feas > max_feas) {
                        max_feas = feas;
                        max_i = i; // Constraint index
                        max_j = j; // Sub constraint index
                    }
                }
            }
        }

        // Apply the weight scaling
        if (max_i >= 0) {
            gparams.all_op[max_i]->weight(max_j) *= gparams.grw_scale;
            // cout << "global rw = " << max_i << " " << max_j << endl;
        }

    }
}

void logger(GroptParams &gparams, int iiter, VectorXd &X, CG_Iter &cg, bool force_print)
{  
    
    if ( ( ((iiter % gparams.verbose_int) == 0) && (gparams.verbose > 1) ) || force_print ){

        double current_obj = 0.0;
        double max_diff = 0.0;    
        // Get the current objective value
        if (gparams.all_obj.size() > 0) {
            current_obj = gparams.all_obj[0]->hist_obj(0,iiter);
            max_diff = 0.0;
            
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
        }


        cout << "~~~~~  Update iiter = " << iiter << "  ~~~~~";
        cout << "   CG iter = " << cg.hist_n_iter[iiter];
        cout << "   current_obj = " << current_obj;
        cout << "   max_diff = " << max_diff;
        cout << "   total_n_feval = " << gparams.total_n_feval << endl;
        
        
        
        for (int i = 0; i < gparams.all_op.size(); i++) {
            cout << "   *** " << gparams.all_op[i]->name
            << "  --  Checks: " << gparams.all_op[i]->hist_check.col(gparams.last_iiter).transpose()
            << "  --  Weights: " << gparams.all_op[i]->weight.transpose()
            << "  --  r_feas: " << gparams.all_op[i]->hist_feas.col(iiter).transpose() << endl;
        }



    }
}

int do_checks(GroptParams &gparams, int iiter, VectorXd &X)
{    

    if (X.hasNaN()) {
        cout << "optimize.cpp stopped due to NaN in X" << endl;
        return 1;
    }



    double current_obj = 0.0;
    double max_diff = 0.0;    
    // Get the current objective value
    if (gparams.all_obj.size() > 0) {
        current_obj = gparams.all_obj[0]->hist_obj(0,iiter);
        max_diff = 0.0;
        
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
    }

    // Sum up all of the check values, (0 is pass so anything greater than 0 does not pass)
    double all_check = 0.0;
    for (int i = 0; i < gparams.all_op.size(); i++) {
        all_check += gparams.all_op[i]->hist_check.col(iiter).sum();
    }
    
    // cout << "    do_check Iter = " << iiter << "  " << current_obj << "  " << max_diff << "  " << all_check << endl;
    
    // Start reducing objective weights if we are close to the end and aren't passing checks
    if ((all_check > 0) && (gparams.obj_min > 0) && (iiter > (gparams.N_iter - gparams.obj_min)))
    {
        cout << "Reducing weights: " ;
        for (int i = 0; i < gparams.all_obj.size(); i++) {
            gparams.all_obj[i]->weight *= gparams.obj_scale;
            cout << gparams.all_obj[i]->weight << " ";
        }
        cout << " Done!" << endl;
    }


    // --- Stopping Checks ---
    if ((max_diff < gparams.d_obj_thresh) && (all_check < 0.5)  && (iiter > 0)) {
        // The waveform has converged and passed all checks
        if (gparams.verbose > 0) {
            cout << "optimize.cpp do_checks() passed!  iiter = " << iiter << "  obj = " << current_obj << "   " << max_diff << "   " << all_check << endl;
        }
        gparams.final_good = 2; // Converged and passed checks
        return 1;
    } else if ((gparams.total_n_feval > (gparams.N_feval - 1000)) && (all_check < 0.5)  && (iiter > 0)) {
        // Waveforms passed checks, but did not converge (and within 1000 f_eval of f_eval limit)
        if (gparams.verbose > 0) {
            cout << "optimize.cpp do_checks() semi-passed!  iiter = " << iiter << "  " << current_obj << "  " 
            << max_diff << "  " << gparams.total_n_feval << "  " << gparams.N_feval << "  " << all_check << endl;
        }
        gparams.final_good = 1; // Passed checks, didn't converge
        return 1;
    } else if (gparams.total_n_feval > gparams.N_feval) {
        // Did too many f_eval and did not converge or pass checks
        if (gparams.verbose > 0) {
            cout << "optimize.cpp do_checks() failed, too many f_eval  iiter = " << iiter << "  " << gparams.total_n_feval << "  " << gparams.N_feval << "  " << all_check << endl;
        }
        gparams.final_good = 0;
        return 1;
    } else if ( (iiter == (gparams.N_iter-1)) && (all_check < 0.5) ) {
        // Waveforms passed checks, but did not converge (and on last iteration)
        if (gparams.verbose > 0) {
            cout << "optimize.cpp do_checks() semi-passed2! last iiter  iiter = " << iiter << "  obj = " << current_obj << "  " << gparams.total_n_feval << "  " << gparams.N_feval << "  " << all_check << endl;
        }
        gparams.final_good = 1;
        return 1;    
    } else {
        gparams.final_good = 0;
        return 0;
    }


}

void optimize(GroptParams &gparams, VectorXd &out)
{
    // Initialize starting vector
    VectorXd X;
    X = gparams.X0;

    MatrixXd Xhist;
    Xhist.setZero(N_HIST_TEMP, gparams.Naxis*gparams.N);

    // Initialize the CG iteration class
    CG_Iter cg(gparams.Naxis*gparams.N, gparams.cg_niter, 
                gparams.cg_resid_tol, gparams.cg_abs_tol);

    for (int i = 0; i < gparams.all_op.size(); i++) {
        gparams.all_op[i]->init(X, gparams.do_init);
    }

    // Number of function evaluations (counts CG iterations)
    gparams.total_n_feval = 0;

    // Actual iterations
    int iiter;
    for (iiter = 0; iiter < gparams.N_iter; ++iiter) {

        // Store what iteration we are on for later array indexing
        gparams.last_iiter = iiter;

        // Do CG iterations
        X = cg.do_CG(gparams.all_op, gparams.all_obj, X);

        gparams.total_n_feval += cg.hist_n_iter[iiter];

        // Update all constraints (do prox operations)        
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

        logger(gparams, iiter, X, cg, false);
    }

    if (gparams.verbose > 0) {
        cout << "!!!!!!!!!!! Final Logger: !!!!!!!!!!!" << endl;
        logger(gparams, iiter-1, X, cg, true);
        cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << endl;
    }
    

    out = X;

    return;
}

// Interp one vector to another based only on its size
// The first and last points match up exactly, and then everything is linearly interpolated
void interp_vec2vec(VectorXd &vec0, VectorXd &vec1) {
    int N0 = vec0.size();
    int N1 = vec1.size();
    
    if (N0 == N1) {
        vec1 = vec0;
        return;
    }

    double tt;
    double di0;
    int i0_lo, i0_hi;
    double v_lo, v_hi;
    double d_lo, d_hi;

    for (int i1 = 0; i1 < N1; i1++) {
        
        tt = (double)i1 / (N1-1);
        
        di0 = tt * (N0-1);
        i0_lo = floor(di0);
        if (i0_lo < 0) {i0_lo = 0;}  // This shouldn't happen unless some weird rounding and floor?
        i0_hi = i0_lo + 1;

        if (i0_hi < N0) {
            d_lo = fabs(di0-i0_hi);
            d_hi = 1.0 - d_lo;

            v_lo = d_lo * vec0(i0_lo);
            v_hi = d_hi * vec0(i0_hi);

            vec1(i1) = v_lo + v_hi;
        } else {
            d_lo = fabs(di0-i0_hi);
            v_lo = d_lo * vec0(i0_lo);
            vec1(i1) = v_lo;
        }
    }
}