#include <iostream> 
#include <string>
#include <math.h>  
#include "Eigen/Dense"

using namespace Eigen;
using namespace std; 

#include "op_btensor.h" 

Op_BTensor::Op_BTensor(int N, int Naxis, double dt) 
    : Operator(N, Naxis, dt, 9, Naxis*N, false)
{
    name = "b-tensor"; 

    do_rw = false;
    balanced = false;
    
    for (int j = 0; j < 9; j++) {
        balance_mod(j) = 1.0;
    }

    GAMMA = 267.5221900e6;  // rad/S/T
    MAT_SCALE = pow((GAMMA / 1000.0 * dt), 2.0) * dt;  // 1/1000 is for m->mm in b-value

    for (int j = 0; j < 9; j++) {
        spec_norm2(j) = (N*N+N)/2.0 * MAT_SCALE;
    }
    
}

void Op_BTensor::set_params(double bval_in)
{
    do_rw = true;

    for (int j = 0; j < Naxis; j++) {
        target(j) = bval_in;
        tol0(j) = 1.0e-1;
        tol(j) = (1.0-cushion) * tol0(j);

        if (balanced) {
            balance_mod(j) = 1.0 / tol(j);
        } else {
            balance_mod(j) = 1.0;
        }
    }

    bval0 = bval_in;

}

void Op_BTensor::set_params(double bval_in0, double bval_in1, double bval_in2)
{
    do_rw = true;
    target.setZero();

    target(0) = bval_in0;
    target(4) = bval_in1;
    target(8) = bval_in2;

    for (int j = 0; j < 9; j++) {
        tol0(j) = 1.0e-1;
        tol(j) = (1.0-cushion) * tol0(j);

        if (balanced) {
            balance_mod(j) = 1.0 / tol(j);
        } else {
            balance_mod(j) = 1.0;
        }
    }

    bval0 = bval_in0;

}


void Op_BTensor::get_feas(VectorXd &s, int iiter)
{
    feas_temp = s;
    prox(feas_temp);
    feas_temp = s - feas_temp;

    Map<Matrix<double, Dynamic, Dynamic, RowMajor>> s_matrix(s.data(), 3, s.size()/3);
    Matrix3d s_tensor = s_matrix*s_matrix.transpose();

    Map<Matrix<double, Dynamic, Dynamic, RowMajor>> feas_matrix(feas_temp.data(), 3, feas_temp.size()/3);
    Matrix3d feas_tensor = feas_matrix*feas_matrix.transpose();


    for (int i = 0; i < 9; i++) {
        int irow = i/3; // Row major
        int icol = i%3;
        
        r_feas(i) = feas_tensor(irow, icol)/s_tensor(irow, icol);
    }

    hist_feas.col(iiter) = r_feas;
}


void Op_BTensor::check(VectorXd &X, int iiter)
{
    Map<Matrix<double, Dynamic, Dynamic, RowMajor>> q3(X.data(), 3, X.size()/3);
    Matrix3d btensor = q3*q3.transpose();

    for (int i = 0; i < 9; i++) {
        int irow = i/3; // Row major
        int icol = i%3;

        double bval_t = btensor(irow, icol);    
        
        feas_check(i) = fabs(bval_t - target(i));

        if (iiter%20 == 0) {
            cout << "   ^^^ bval_t " << iiter << "  " << bval_t << "  " << feas_check(i) << "  " << target(i) << "  " << tol0(i) << "  " << balance_mod(i) << " " << r_feas(i) << endl;
        }
    }

    for (int i = 0; i < feas_check.size(); i++) {
        if (feas_check[i] > tol0[i]) {
            hist_check(i, iiter) = 1.0;
        } else {
            hist_check(i, iiter) = 0.0;
        }
    }
}

void Op_BTensor::forward(VectorXd &X, VectorXd &out, 
                         bool apply_weight, int norm, bool no_balance)
{
    for (int j = 0; j < Naxis; j++) {
        int jN = j*N;
        double gt = 0;    
        for (int i = 0; i < N; i++) {
            gt += X(jN + i) * inv_vec(jN + i);
            out(jN + i) = gt * sqrt(MAT_SCALE);
        }
    }

    if (apply_weight) {
        out.array() *= weight(0);
    }

    if (balanced && !no_balance) {
        out.array() *= balance_mod(0);
    }
}


void Op_BTensor::transpose(VectorXd &X, VectorXd &out, 
                           bool apply_weight, int norm)
{
    for (int j = 0; j < Naxis; j++) {
        int jN = j*N;
        double gt = 0;    
        for (int i = N-1; i >= 0; i--) {
            gt += X(jN + i) * sqrt(MAT_SCALE);
            out(jN + i) = gt * inv_vec(jN + i);
        }
    }

    if (balanced) {
        out.array() /= balance_mod(0);
    }

    if (apply_weight) {
        out.array() *= weight(0);
    }

    if (norm == 2) {
        out.array() /= spec_norm2(0);
    }

    out.array() *= fixer.array();
}


void Op_BTensor::prox(VectorXd &X)
{

    Map<Matrix<double, Dynamic, Dynamic, RowMajor>> q3(X.data(), 3, X.size()/3);

    for (int jj = 0; jj < 1; jj++) {
        
        Matrix3d btensor = q3*q3.transpose();
        
        Matrix3d btensor0;
        for (int i = 0; i < 9; i++) {
            int irow = i/3; // Row major
            int icol = i%3;

            double min_val = balance_mod(i) * (target(i) - tol(i));
            double max_val = balance_mod(i) * (target(i) + tol(i));
            double val = btensor(irow, icol);

            if (val < min_val) {
                btensor0(irow, icol) = min_val;
            } else if (val > max_val) {
                btensor0(irow, icol) = max_val;
            } else {
                btensor0(irow, icol) = val;
            }
        }


        Matrix3d pinv = btensor.completeOrthogonalDecomposition().pseudoInverse();
        Matrix3d mod = btensor0 * pinv;

        JacobiSVD<Matrix3d> svd(mod, ComputeThinU | ComputeThinV);
        Vector3d s2 = svd.singularValues().cwiseSqrt();

        Matrix3d mod2 = svd.matrixU() * s2.asDiagonal() * svd.matrixV().adjoint();

        q3 = mod2 * q3;
    }
}

void Op_BTensor::get_obj(VectorXd &X, int iiter)
{
    Ax_temp.setZero();
    forward(X, Ax_temp, false, 0, true);
    current_obj = Ax_temp.squaredNorm();
    hist_obj(0, iiter) = current_obj;
    
}


void Op_BTensor::test(VectorXd &X)
{
    VectorXd target_temp(9);
    target_temp << 200, 10, 0, 0, 200, 0, 0, 0, 200;

    VectorXd tol_temp(9);
    tol_temp.setOnes();
    tol_temp *= 1.0e-2;
    
    
    Map<Matrix<double, Dynamic, Dynamic, RowMajor>> g3(X.data(), 3, X.size()/3);

    // cout << "g3: " << endl << g3.transpose() << endl << endl;

    // cout << "----------------------" << endl;
    
    
    Ax_temp.setZero();



    forward(X, Ax_temp, false, 0, true);
    
    Map<Matrix<double, Dynamic, Dynamic, RowMajor>> q3(Ax_temp.data(), 3, Ax_temp.size()/3);

    // cout << "q3: " << endl << Ax_temp.size() << endl << q3.transpose() << endl << endl;
    // cout << q3.rows() << endl;
    // cout << q3.cols() << endl;

    Matrix3d btensor = q3*q3.transpose();
    Matrix3d btensor0;
    for (int i = 0; i < 9; i++) {
        int irow = i/3; // Row major
        int icol = i%3;


        double min_val = target_temp(i) - tol_temp(i);
        double max_val = target_temp(i) + tol_temp(i);
        double val = btensor(irow, icol);

        if (val < min_val) {
            btensor0(irow, icol) = min_val;
        } else if (val > max_val) {
            btensor0(irow, icol) = max_val;
        } else {
            btensor0(irow, icol) = val;
        }


    }


    
    Matrix3d pinv = btensor.completeOrthogonalDecomposition().pseudoInverse();
    Matrix3d mod = btensor0 * pinv;

    // cout << "btensor0: " << endl << btensor0 << endl << endl;
    cout << "btensor: " << endl << btensor << endl << endl;
    // cout << "pinv: " << endl << pinv << endl << endl;
    // cout << "mod: " << endl << mod << endl << endl;

    JacobiSVD<Matrix3d> svd(mod, ComputeThinU | ComputeThinV);
    Vector3d s2 = svd.singularValues().cwiseSqrt();
    // cout << "s: " << endl << svd.singularValues() << endl << endl;
    // cout << "s2: " << endl << s2 << endl << endl;

    Matrix3d mod2 = svd.matrixU() * s2.asDiagonal() * svd.matrixV().adjoint();

    // cout << "mod2: " << endl << mod2 << endl << endl;

    // g3 = mod2 * g3;

}