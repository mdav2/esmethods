#ifndef __HF_H__
#define __HF_H__
#include <iostream>
#include <memory>
#include <string>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <gsl/gsl_eigen.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_blas.h>

namespace HF {
class tensor4p
{
    public:
        tensor4p (int parraysize);
        ~tensor4p ();
        std::unique_ptr<double*** []> myarray;
        int arraysize;
};

class wfn
{
    private:
        //>Private Data
        gsl_matrix * evec; //= gsl_matrix_alloc (nbf, nbf);
        gsl_matrix * lambda; //= gsl_matrix_alloc (nbf, nbf);
        gsl_matrix * lambdapf; //= gsl_matrix_alloc (nbf, nbf);
        gsl_matrix * F; //= gsl_matrix_alloc (nbf, nbf);
        gsl_matrix * FPROTECT;
        gsl_matrix * Fp; // = gsl_matrix_alloc (nbf, nbf);
        gsl_matrix * Hcore;  //= gsl_matrix_alloc (nbf, nbf);
        gsl_eigen_symmv_workspace * w; //= gsl_eigen_symmv_alloc (nbf);
        //<Private Data

        //>Private Methods
        void read_enuc (std::string fname);
        void read_2D (std::string fname, gsl_matrix * A);
        void read_ERI (std::string fname);
        void build_D (void);
        void build_F (void);
        void AOtoMOnoddy (void);
        //<Private Methods

    public:
        //>Public Data
        tensor4p * AO_eri;
        tensor4p * MO_eri;
        gsl_vector * eval; //= gsl_vector_alloc (5);// = gsl_vector_alloc (nbf);
        gsl_matrix * C;
        gsl_matrix * AO_overlap; //= gsl_matrix_alloc (nbf, nbf);
        gsl_matrix * AO_potential; //= gsl_matrix_alloc (nbf, nbf);
        gsl_matrix * AO_kinetic; //= gsl_matrix_alloc (nbf, nbf);
        gsl_matrix * D ; //= gsl_matrix_alloc (nbf, nbf);
        int natom;
        int nbf; 
        int nocc;
        int maxiter;
        double E;
        double Ecorr;
        double enuc;
        //<Public Data

        //>Public Methods
        void do_MP2 (void);
        double do_SCF (void);
        double compute_E (void);
        wfn(int nbf);
        ~wfn(void);
        //<Public Methods
};
}
#endif
