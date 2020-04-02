#include "optimize.h"
#include "cg_iter.h"

using namespace Eigen;
using namespace std;

#define N_HIST_TEMP 20

int do_checks(GroptParams &gparams, int iiter, VectorXd &X)
{
    double current_obj = gparams.all_obj[0]->hist_obj(0,iiter);
    double max_diff = 0.0;
    
    for (int i = 0; i < N_HIST_TEMP; i++) {
        if ((iiter - i) < 0) {break;}
        double obj = gparams.all_obj[0]->hist_obj(0,iiter - i);
        double diff = fabs(current_obj - obj);
        if (diff > max_diff) {
            max_diff = diff;
        }
    }
    max_diff /= current_obj;

    double all_check = 0.0;
    for (int i = 0; i < gparams.all_op.size(); i++) {
        all_check += gparams.all_op[i]->hist_check.col(iiter).sum();
    }
    
    // if (iiter%20 == 0) {
    //     cout << "do_check Iter = " << iiter << "  " << current_obj << "  " << max_diff << "  " << all_check << endl;         
    // }

    if ((max_diff < 0.001) && (all_check < 0.5)) {
        cout << "Done do_check Iter = " << iiter << "  " << current_obj << "  " << max_diff << "  " << all_check << endl;
        return 1;
    } else {
        return 0;
    }
}

void optimize(GroptParams &gparams, VectorXd &out)
{
    VectorXd X;
    X = gparams.X0;

    CG_Iter cg(gparams.N, gparams.cg_niter, 
                gparams.cg_resid_tol, gparams.cg_abs_tol);


    for (int i = 0; i < gparams.all_op.size(); i++) {
        gparams.all_op[i]->prep_y(X);
    }

    for (int iiter = 0; iiter < gparams.N_iter; iiter++) {
        X = cg.do_CG(gparams.all_op, gparams.all_obj, X);
                
        for (int i = 0; i < gparams.all_op.size(); i++) {
            gparams.all_op[i]->update(X, iiter);
        }
        
        for (int i = 0; i < gparams.all_obj.size(); i++) {
            gparams.all_obj[i]->get_obj(X, iiter);
        }

        if (do_checks(gparams, iiter, X) > 0) {
            break;
        }
    }

    out = X;

    return;
}

OptHist::OptHist(GroptParams &gparams) 
{

}