#ifndef __phf_H__
#define __phf_H__
#include <iostream>
#include <memory>
#include <string>
#include <fstream>
#include <iomanip>
#include <cmath>
//#include <gsl/gsl_eigen.h>
//#include <gsl/gsl_math.h>
//#include <gsl/gsl_blas.h>

namespace psi{ namespace phf {
double kron (int p, int q); 
class tensor4 //class to store 4D tensors and manage memory.
//Access elements via varname->myarray[i][j][k][l]
{
    private:
        SharedMatrix myarray;
        int ijtoa (int i, int j) {
            return i*this->arraysize + j;
        }
    public:
        tensor4 (SharedMatrix, int parraysize);
        tensor4 (int parraysize);
        ~tensor4 ();
        void print (int l1, int u1, int l2, int u2, int l3, int u3, int l4, int u4) {
            for ( int i = l1; i < u1; i++ ) {
                for ( int j = l2; j < u2; j++ ) {
                    for ( int l = l3; l < u3; l++ ) {
                        for ( int k = l4; k < u4; k++ ) {
                            std::cout << std::setprecision(6) << this->get(i,j,k,l) << " "; 
                        }
                        std::cout << "\n";
                    }
                    std::cout << "\n\n";
                }
            }
        }
        double get (int i, int j, int k, int l) {
            //Retrieve a value from the tensor
            int a = this->ijtoa(i,j);
            int b = this->ijtoa(k,l);;
            return this->myarray->get(a,b);
        }
        void set (int i, int j, int k, int l, double val) {
            //Set a value in the tensor
            int a = this->ijtoa(i, j);
            int b = this->ijtoa(k,l);
            this->myarray->set(a,b,val);
        }
        int arraysize;
};

class phfwfn
//class to perform SCF procedure, integral transformations, and
//post HF (MP2 and CCSD) energy computations.
{
    private:
        //>Private Data
        MintsHelper * mints;
        Vector * eval;
        SharedMatrix C;
        SharedMatrix FSO;
        SharedMatrix Hcore;
        SharedMatrix HcoreSO;
        SharedMatrix FF;
        SharedMatrix Dia;
        SharedMatrix tia;
        SharedMatrix tia_new;
        tensor4 * SO_eri;
        tensor4 * MO_eri;
        tensor4 * AO_eri;
        tensor4 * W;
        tensor4 * tautijab;
        tensor4 * tauijab;
        tensor4 * tijab;
        tensor4 * tijab_new;
        tensor4 * Dijab;

       // //>Private Methods
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
        //void print_Fp (void);
        void print2D (SharedMatrix A, int lb1, int ub1, int lb2, int ub2);
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
        void build_tijab_MP2 (void);
        double MP2viaCC (void);
        void MOtoSO (void);
        void FtoSO (void);
        void HcoretoSO (void);
        //void print_MO_F (void);
        //void print_MO_F_alt (void);
        //<Private Methods

    public:
        //>Public Data
        // tensor4p * AO_eri;
        // tensor4p * MO_eri;
        // tensor4p * SO_eri;
        int natom;
        int nbf; 
        int nmo;
        int nocc;
        int noccso;
        int maxiter;
        double E;
        double Ecorr;
        double enuc;
        // //<Public Data

        // //>Public Methods
        // void do_CCSD (void);
        void do_MP2 (void);
        void do_CCSD (void);
        // double do_SCF (void);
        // double compute_E (void);
        phfwfn(SharedWavefunction ref_wfn, int nbf, int nocc);
        ~phfwfn(void);
        // //<Public Methods
};
}}
#endif
