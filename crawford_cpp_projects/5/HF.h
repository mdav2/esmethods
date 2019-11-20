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
double kron (int p, int q); class tensor4p //class to store 4D tensors and manage memory.
//Access elements via varname->myarray[i][j][k][l]
{
    public:
        tensor4p (int parraysize);
        ~tensor4p ();
        std::unique_ptr<double*** []> myarray;
        int arraysize;
};

class wfn
//class to perform SCF procedure, integral transformations, and
//post HF (MP2 and CCSD) energy computations.
{
    private:
        //>Private Data
        gsl_matrix * evec; //= gsl_matrix_alloc (nbf, nbf);
        gsl_matrix * lambda; //= gsl_matrix_alloc (nbf, nbf);
        gsl_matrix * lambdapf; //= gsl_matrix_alloc (nbf, nbf);
        gsl_matrix * FPROTECT; //used for copy protecting during eigensolve
        gsl_matrix * FSO; //F in SO basis
        gsl_matrix * Hcore;  //= gsl_matrix_alloc (nbf, nbf);
        gsl_matrix * HcoreSO;  //= gsl_matrix_alloc (nbf, nbf);
        gsl_matrix * FF;
        gsl_matrix * Dia;
        gsl_matrix * tia;
        gsl_matrix * tia_new;
        tensor4p * W;
        tensor4p * tautijab;
        tensor4p * tauijab;
        tensor4p * tijab;
        tensor4p * tijab_new;
        tensor4p * Dijab;
        gsl_eigen_symmv_workspace * w; //= gsl_eigen_symmv_alloc (nbf);
        //<Private Data

        //>Private Methods
        void read_enuc (std::string fname);
        void read_2D (std::string fname, gsl_matrix * A);
        void read_ERI (std::string fname);
        void build_D (void); //D for HF theory
        void build_F (void);
        void cciter (void);
        void ccenergy (void);
        void build_Wmnij (void);
        void build_Wabef (void);
        void build_Wmbej (void);
        void build_Fae (void);
        void build_Fmi (void);
        void build_Fme (void);
        void build_tautijab (void);
        void build_tauijab (void);
        void build_Dia (void); //D for CC intermediates
        void build_Dijab (void);
        void build_tia (void);
        void tiacpy (void);
        void build_tijab (void);
        void tijabcpy (void);
        void build_tijab_MP2 (void);
        //void print_Fp (void);
        void print_F (void);
        void print_Fae (void);
        void print_Fme (void);
        void print_Fmi (void);
        void print_Wabef (void);
        void print_Wmnij (void);
        void print_Wmbej (void);
        void print_tia (void);
        void print_tijab (void);
        void print_tauijab (void);
        void print_tautijab (void);
        void print_MO_F (void);
        void print_MO_F_alt (void);
        void AOtoMOnoddy (void);
        void MOtoSO (void);
        void FtoSO (void);
        void HcoretoSO (void);
        double MP2viaCC (void);
        //<Private Methods

    public:
        //>Public Data
        tensor4p * AO_eri;
        tensor4p * MO_eri;
        tensor4p * SO_eri;
        gsl_vector * eval; //= gsl_vector_alloc (5);// = gsl_vector_alloc (nbf);
        gsl_matrix * C;
        gsl_matrix * F; //= gsl_matrix_alloc (nbf, nbf);
        gsl_matrix * Fp; // = gsl_matrix_alloc (nbf, nbf);
        gsl_matrix * AO_overlap; //= gsl_matrix_alloc (nbf, nbf);
        gsl_matrix * AO_potential; //= gsl_matrix_alloc (nbf, nbf);
        gsl_matrix * AO_kinetic; //= gsl_matrix_alloc (nbf, nbf);
        gsl_matrix * D ; //= gsl_matrix_alloc (nbf, nbf);
        int natom;
        int nbf; 
        int nmo;
        int nocc;
        int noccso;
        int maxiter;
        double E;
        double Ecorr;
        double enuc;
        //<Public Data

        //>Public Methods
        void do_CCSD (void);
        void do_MP2 (void);
        double do_SCF (void);
        double compute_E (void);
        wfn(int nbf, int nocc, int natom);
        ~wfn(void);
        //<Public Methods
};
}
#endif
