#include <iostream>
#include <string>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <gsl/gsl_eigen.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_blas.h>
#include "HF.h"

int main() {
    int nbf = 7;
    HF::wfn mywfn(nbf);
    mywfn.do_SCF ();
    std::cout << std::setprecision(15) << "E[SCF] " << mywfn.E << "\n";
    //for (int i = 0;  i < mywfn.nbf; i++) {
    //    for (int j = 0; j < mywfn.nbf; j++) {
    //        std::cout << gsl_matrix_get ( mywfn.F, i, j ) << " ";
    //    }
    //    std::cout << "\n"; 
    //}
    mywfn.do_MP2 ();
    std::cout << std::setprecision(15) << "E[cor] " << mywfn.Ecorr << "\n";
    std::cout << std::setprecision(15) << "E[MP2] " << mywfn.E + mywfn.Ecorr << "\n";
    mywfn.do_CCSD ();
    return 0;
}
