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
    HF::wfn mywfn(7);
    mywfn.do_SCF ();
    mywfn.do_MP2 ();
    std::cout << std::setprecision(15) << "E[SCF] " << mywfn.E << "\n";
    std::cout << std::setprecision(15) << "E[cor] " << mywfn.Ecorr << "\n";
    std::cout << std::setprecision(15) << "E[MP2] " << mywfn.E + mywfn.Ecorr << "\n";
    return 0;
}
