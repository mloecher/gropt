#include <iostream> 
#include <string>
#include <math.h>  
#include "Eigen/Dense"

using namespace Eigen;
using namespace std; 

#include "wrappers.h"
#include "scratch.h"

void svd_test()
{
    cout << "In SVD test . . .  " << endl;
    MatrixXd btensor0(3,3); 
    btensor0 << 200, 100, 0,
                50, 150, 0,
                0, 0, 0;

    cout << endl << "btensor0 = " << endl << btensor0 << endl << endl;

    JacobiSVD<MatrixXd> svd0(btensor0, ComputeThinU | ComputeThinV);

    MatrixXd btensor_svd;
    btensor_svd = svd0.matrixU() * svd0.singularValues().asDiagonal() * svd0.matrixV().transpose();

    cout << endl << "btensor_svd = " << endl << btensor_svd << endl << endl;

    MatrixXd btensor_sqrt;
    VectorXd singvals0_sq = svd0.singularValues().cwiseSqrt();

    btensor_sqrt = svd0.matrixU() * singvals0_sq.asDiagonal() * svd0.matrixV().transpose();

    cout << endl << "svd0.singularValues() = " << endl << svd0.singularValues() << endl << endl;
    cout << endl << "singvals0_sq = " << endl << singvals0_sq << endl << endl;

    cout << endl << "btensor_sqrt = " << endl << btensor_sqrt << endl << endl;
    cout << endl << "btensor_sqrt^2 = " << endl << btensor_sqrt*btensor_sqrt << endl << endl;

}   


int main()
{
    svd_test();
    // oned_flowcomp();

    // threed_flowcomp();

    // diff_duty_cycle();
    // -------------------------------------
    // double *params0; 
    // double *params1; 
    // double *out0;
    // double *out1; 
    // double *out2; 
    // int *outsize;
    // python_wrapper_warmstart_v1(params0, params1, &out0, &out1, &out2, &outsize);
    

    // -------------------------------------
    // double dt0 = 400.0e-6;
    // double dt_out = 100e-6;
    // double T_90 = 4.0e-3;
    // double T_180 = 6.0e-3;
    // double T_readout = 12.0e-3;
    // double TE = 64.0e-3;

    // int MMT = 2;

    // double gmax = .04;
    // double smax = 50.0;

    // int verbose = 0;

    // double *G_out;
    // int N_out;
    

    // gropt_diff_seq(&G_out, &N_out,  verbose,
    //                 dt0, dt_out, gmax, smax, TE,
    //                 T_readout, T_90, T_180, MMT);
    return 0;
}