#include <iostream> 
#include <string>
#include <math.h>  
#include <Eigen/Dense>
#include <vector>  

using namespace Eigen;
using namespace std; 

#include "op_main.h"
#include "cg_iter.h"

void get_Ax(vector<Operator*> all_op, vector<Operator*> all_obj, VectorXd &X, VectorXd &Ax)
{
    for (int i = 0; i < all_op.size(); i++) {
        all_op[i]->add2AtAx(X, Ax);
    }

    for (int i = 0; i < all_obj.size(); i++) {
        all_obj[i]->add2AtAx(X, Ax);
    }
}

CG_Iter::CG_Iter(int N, int max_iter, double resid_tol, double abs_tol) 
    : N(N), max_iter(max_iter), resid_tol(resid_tol), abs_tol(abs_tol)
{
    b.setZero(N);
    Ax.setZero(N);
    Ap.setZero(N);
    x1.setZero(N);
    r.setZero(N);
    p.setZero(N); 
}

VectorXd CG_Iter::do_CG(vector<Operator*> all_op, vector<Operator*> all_obj, VectorXd &x0)
{
    double rnorm0;
    double rnorm1;
    double r_k_norm;
    double r_kplus1_norm;
    double pAp;
    double alpha;
    double beta;
    
    Ax.setZero();
    Ap.setZero();
    x1 = x0;

    b.setZero();
    for (int i = 0; i < all_op.size(); i++) {
        all_op[i]->add2b(b);
    }

    get_Ax(all_op, all_obj, x1, Ax);
    r = (Ax - b);

    rnorm0 = r.norm();
    
    p = -r;
    r_k_norm = r.dot(r);

    int ii;
    for (ii = 0; ii < max_iter; ii++) {
        Ap.setZero();
        
        get_Ax(all_op, all_obj, p, Ap);
        pAp = p.dot(Ap);
        // cout << "    cg iter " << ii << " pAp = " << pAp << endl;
        if (pAp < abs_tol) {break;}
        
        alpha = r_k_norm / pAp;
        x1 += alpha * p;
        r += alpha * Ap;

        r_kplus1_norm = r.dot(r);
        beta = r_kplus1_norm / r_k_norm;
        r_k_norm = r_kplus1_norm;
        rnorm1 = sqrt(r_k_norm);
        // cout << "    cg iter " << ii << " rnorm1 = " << rnorm1 << endl;
        if ( (rnorm1/rnorm0) < resid_tol ) {break;}
        if ( r_k_norm < abs_tol ) {break;}

        p = beta * p - r;
    }

    n_iter = ii+1;
    return x1;
}

int CG_Iter::get_n_iter()
{
    return n_iter;
}