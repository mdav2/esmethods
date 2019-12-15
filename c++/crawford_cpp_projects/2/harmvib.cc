#include <iostream>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <gsl/gsl_eigen.h>
#include <gsl/gsl_math.h>


using namespace std;
int main () {
    int natom;
    int natom2;
    //>open geometry and hess, read in natoms
    ifstream geom("h2o.geom");
    ifstream hess("h2o.hess");
    geom >> natom; //get number of atoms
    geom.close();
    hess >> natom2;

    if ( natom != natom2 ) {
        cout << "natom from geometry and hessian don't match";
        return 1;
    }
    //<open geometry and hess, read in natoms

    //>Allocate memory
    int *zval = new int[natom];
    double *mass = new double[natom];
    gsl_matrix * F = gsl_matrix_alloc (3*natom, 3*natom); 
    gsl_matrix_set_zero (F);
    gsl_vector *eval = gsl_vector_alloc (3*natom);
    gsl_eigen_symm_workspace * w = gsl_eigen_symm_alloc (3*natom);
    //<Allocate memory

    //>masses hardcoded
    mass[0] = 15.99491461957;
    mass[1] = 1.00782503223;
    mass[2] = 1.00782503223;
    //<masses hardcoded

    //>read in hessian
    double *tmp = new double[3];
    for(int i=0; i < 3*natom; i++) {
        for(int j=0; j < natom; j++) {
            hess >> tmp[0]
                 >> tmp[1] 
                 >> tmp[2];
            gsl_matrix_set (F,i,3*j+0, tmp[0]);
            gsl_matrix_set (F,i,3*j+1, tmp[1]);
            gsl_matrix_set (F,i,3*j+2, tmp[2]);
        }
    }
    hess.close();
    //<read in hessian


    //> mass weight hessian
    double den;
    int i_,j_;
    for(int i = 0; i < 3*natom; i++) {
        for(int j = 0; j < 3*natom; j++) {
            i_ = floor (i/3);
            j_ = floor (j/3); 
            den = sqrt(mass[i_]*mass[j_]);
            gsl_matrix_set (F,i,j, gsl_matrix_get (F,i,j)/den);
        }
    }
    //< mass weight hessian

    //>Diagonalize hessian
    gsl_eigen_symm (F, eval, w);
    //<Diagonalize hessian

    //>Print output
    cout << "Printing eigenvalues of H_mw and frequencies (cm^-1)" << endl;
    double etmp;
    for (int i = 0; i < 3*natom; i++) {
        etmp = gsl_vector_get (eval, i);
        if (fabs(etmp) < 1E-14) {
            etmp = 0.0;
        }
        gsl_vector_set (eval,i,sqrt(etmp));
        cout << etmp << " " << 5140.485*sqrt(etmp) << endl;
    }
    //<Print output
    
    //>Deallocate memory
    delete[] zval;
    delete[] mass;
    delete[] tmp;
    gsl_matrix_free (F);
    gsl_vector_free (eval);
    gsl_eigen_symm_free (w);
    //<Deallocate memory
}

