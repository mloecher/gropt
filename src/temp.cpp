#include <iostream>
#include <vector>
#include "Eigen/Dense"
#include "op_main.h"
#include "op_moments.h"
#include "op_bval.h"
#include "op_slew.h"
#include "op_gradient.h"
#include "cg_iter.h"
#include <typeinfo>
#include <chrono> 

using namespace Eigen;
using namespace std;

void optimize_diff()
{
    double dt = 100e-6;
    double T_90 = 4e-3; 
    double T_180 = 6e-3; 
    double T_readout = 12e-3; 
    double TE = 64e-3;

    int N = (int)((TE-T_readout)/dt) + 1;
    
    int ind_inv = (int)(TE/2.0/dt);
    VectorXd inv_vec;
    inv_vec.setOnes(N);
    for(int i = ind_inv; i < N; i++) {
        inv_vec(i) = -1.0;
    }

    int ind_90_end = ceil(T_90/dt);
    int ind_180_start = floor((TE/2.0 - T_180/2.0)/dt);
    int ind_180_end = ceil((TE/2.0 + T_180/2.0)/dt);

    VectorXd set_vals;
    set_vals.setOnes(N);
    set_vals.array() *= -9999999.0;
    for(int i = 0; i <= ind_90_end; i++) {
        set_vals(i) = 0.0;
    }
    for(int i = ind_180_start; i <= ind_180_end; i++) {
        set_vals(i) = 0.0;
    }
    set_vals(0) = 0.0;
    set_vals(N-1) = 0.0;


    VectorXd fixer;
    fixer.setOnes(N);
    for(int i = 0; i < N; i++) {
        if (set_vals(i) > -10000) {
            fixer(i) = 0.0;
        }
    }

    cout << "N = " << N << endl << endl;
    cout << "ind_inv = " << ind_inv << "  ind_90_end = " << ind_90_end << "  ind_180_start = " << ind_180_start << "  ind_180_end = " << ind_180_end << endl << endl;
    // cout << "inv_vec = " << endl << inv_vec.transpose() << endl << endl;
    // cout << "set_vals = " << endl << set_vals.transpose() << endl << endl;
    // cout << "fixer = " << endl << fixer.transpose() << endl << endl;

    Op_Moments opM(N, dt, 2);
    Op_Slew opS(N, dt);
    Op_Gradient opG(N, dt);
    Op_BVal opB(N, dt);

    opM.set_inv_vec(inv_vec);
    opM.set_fixer(fixer);
    opS.set_inv_vec(inv_vec);
    opS.set_fixer(fixer);
    opG.set_inv_vec(inv_vec);
    opG.set_fixer(fixer);
    opB.set_inv_vec(inv_vec);
    opB.set_fixer(fixer);

    MatrixXd moments(2,7);
    moments << 0, 0, 0, 0, 0, 0.0, 1e-6,
               0, 1, 0, 0, 0, 0.0, 1e-6;
    opM.set_params(moments);
    opS.set_params(100.0);
    opG.set_params(0.05, set_vals);
    opB.weight(0) = -1.0;

    vector<Operator*> all_op;
    all_op.push_back(&opG);
    all_op.push_back(&opS);
    all_op.push_back(&opM);

    vector<Operator*> all_obj;
    all_obj.push_back(&opB);

    VectorXd X;
    X.setOnes(N);
    X.array() *= inv_vec.array() * fixer.array() * .025;

    // cout << "X = " << endl << X.transpose() << endl << endl;
    // cout << "opM.A = " << endl << opM.A << endl << endl;

    CG_Iter cg(N, 1000, 1.0e-2, 1.0e-16);

    auto t_start = chrono::steady_clock::now(); 

    for (int i = 0; i < all_op.size(); i++) {
        all_op[i]->prep_y(X);
    }

    int Niter = 300;
    for (int iiter = 0; iiter < Niter; iiter++) {
        X = cg.do_CG(all_op, all_obj, X);
        // cout << "  CG n_iter = " << cg.n_iter << endl;
        
        for (int i = 0; i < all_op.size(); i++) {
            all_op[i]->update(X, iiter);
        }
    }

    auto t_end = chrono::steady_clock::now(); 
    // Calculating total time taken by the program. 
    double time_taken = chrono::duration_cast<chrono::microseconds>(t_end - t_start).count(); 
  
    cout << "Time taken by program is : " << time_taken << " microsec" << endl; 

    cout << "***Final Res***" << endl;
    cout << "X = " << endl << X.transpose() << endl << endl;

    for (int i = 0; i < all_op.size(); i++) {
        VectorXd temp;
        temp.setZero(all_op[i]->Y0.size());

        all_op[i]->forward(X, temp, false, 0, true);

        cout << all_op[i]->name <<" = " << endl << temp.transpose() << endl << endl;
    }

    VectorXd temp;
    temp.setZero(all_obj[0]->Y0.size());
    all_obj[0]->forward(X, temp, false, 0, true);
    double bval = temp.squaredNorm();

    cout << "bval = " << bval << endl << endl;

}

void optimize()
{   
    double T = 1.3e-3;
    int N = 64;
    double dt = T/(N-1);

    
    Op_Moments opM(N, dt, 2);
    
    MatrixXd moments(2,7);
    moments << 0, 0, 0, 0, 0, 0.0, 1e-6,
               0, 1, 0, 0, 0, 5.3, 1e-6;

    opM.set_params(moments);

    Op_Slew opS(N, dt);
    opS.set_params(100.0);
    Op_Gradient opG(N, dt);
    opG.set_params(0.05);

    vector<Operator*> all_op;
    all_op.push_back(&opG);
    all_op.push_back(&opS);
    all_op.push_back(&opM);

    vector<Operator*> all_obj;
    
    VectorXd X;
    X.setOnes(N);
    X.array() *= opG.fixer.array() * .01;

    CG_Iter cg(N, 1000, 1.0e-3, 1.0e-16);

    auto t_start = chrono::steady_clock::now(); 

    for (int i = 0; i < all_op.size(); i++) {
        all_op[i]->prep_y(X);
    }

    int Niter = 30;
    for (int iiter = 0; iiter < Niter; iiter++) {
        X = cg.do_CG(all_op, all_obj, X);
        cout << "  CG n_iter = " << cg.n_iter << endl;
        
        for (int i = 0; i < all_op.size(); i++) {
            all_op[i]->update(X, iiter);
        }
    }

    auto t_end = chrono::steady_clock::now(); 
    // Calculating total time taken by the program. 
    double time_taken = chrono::duration_cast<chrono::microseconds>(t_end - t_start).count(); 
  
    cout << "Time taken by program is : " << time_taken << " microsec" << endl; 

    cout << "***Final Res***" << endl;
    cout << "X = " << endl << X.transpose() << endl << endl;

    for (int i = 0; i < all_op.size(); i++) {
        VectorXd temp;
        temp.setZero(all_op[i]->Y0.size());

        all_op[i]->forward(X, temp, false, 0, true);

        cout << all_op[i]->name <<" = " << endl << temp.transpose() << endl << endl;
    }

    return;
}

int main()
{
    // MatrixXd m(2,2);
    // m(0,0) = 3;
    // m(1,0) = 2.5;
    // m(0,1) = -1;
    // m(1,1) = m(1,0) + m(0,1);
    // std::cout << m << std::endl;


    // Matrix<int, 3, 4, ColMajor> Acolmajor;
    // Acolmajor << 8, 2, 2, 9,
    //             9, 1, 4, 4,
    //             3, 5, 4, 5;
    // cout << "The matrix A:" << endl;
    // cout << Acolmajor << endl << endl; 
    // cout << "In memory (column-major):" << endl;
    // for (int i = 0; i < Acolmajor.size(); i++)
    //     cout << *(Acolmajor.data() + i) << "  ";

    // cout << endl << endl;

    // for (int i = 0; i < Acolmajor.rows(); i++) {
    //     for (int j = 0; j < Acolmajor.cols(); j++) {
    //         cout << Acolmajor(i,j) << "  ";
    //     }
    // }

    // cout << endl << endl;


    // MatrixXd m;
    // m.resize(3,6);
    // for (int i = 0; i < m.rows(); i++) {
    //     for (int j = 0; j < m.cols(); j++) {
    //         m(i,j) = i;
    //     }
    // }
    // cout << m << endl << endl;

    // RowVectorXd v;
    // v.resize(6);
    // for (int i = 0; i < v.size(); i++) {
    //     v(i) = 2.0;
    // }
    // cout << v << endl << endl;

    // cout << (m*v) << endl << endl;
    
    // VectorXd a(5);
    // VectorXd b(5);

    // for (int i = 0; i < a.size(); i++) {
    //     a(i) = i;
    //     b(i) = i + 5;
    // }
    // cout << a << endl << endl;
    // cout << b << endl << endl;
    // b = a;
    // cout << a << endl << endl;
    // cout << b << endl << endl;
    // a(3) = 100.0;
    // cout << a << endl << endl;
    // cout << b << endl << endl;


    // VectorXd a(5);
    // VectorXd b(5);
    // VectorXd c(5);
    // ArrayXd d(5);

    // for (int i = 0; i < a.size(); i++) {
    //     a(i) = i;
    //     b(i) = i + 5;
    // }
    // cout << a << endl << endl;
    // cout << b << endl << endl;

    // a.array() *= b.array();

    // cout << a << endl << endl;
    // cout << b << endl << endl;

    // cout << a.array()*b.array() << endl << endl;

    // c = a.array()*b.array();

    // cout << typeid(a).name() << endl << endl;
    // cout << typeid(c).name() << endl << endl;
    // cout << typeid(d).name() << endl << endl;



    // VectorXd a(5);
    // VectorXd b(1);
    // b.setOnes();
    // b *= 2.0;
    // for (int i = 0; i < a.size(); i++) {
    //     a(i) = i;
    // }

    // cout << a << endl << endl;
    // cout << b << endl << endl;

    // cout << 1.0 - a << endl << endl;


    // optimize(); 

    optimize_diff(); 

    return 0;

}