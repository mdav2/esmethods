#ifndef __phf_H__
#define __phf_H__
#include <iostream>
#include <memory>
#include <string>
#include <fstream>
#include <iomanip>
#include <cmath>
#include "psi4/libmints/matrix.h"
#include "psi4/libmints/vector.h"
#include "psi4/libmints/mintshelper.h"
//#include <gsl/gsl_eigen.h>
//#include <gsl/gsl_math.h>
//#include <gsl/gsl_blas.h>

namespace psi{ namespace phf {
double kron (int p, int q); 
class tensor4 //class to store 4D tensors and manage memory.
//Access elements via varname->get<set>(i,j,k,l,<val>)
{
    private:
        inline int ijtoa (int i, int j) {
            return i*this->arraysize + j;
        }
    public:
        SharedMatrix myarray;
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

        //These slice functions produce a 2D matrix for gemms etc
        //via constant e.g. i and j -> matrix of k,l

        double** get_pointer(int h) {
            double** pointy = this->myarray->pointer(h);
            return pointy;
        }
        double** sliceij (int i, int j) {
            const int sz = this->arraysize;
            //SharedMatrix slice = Matrix(this->arraysize,this->arraysize).clone();
            //for ( int k = 0; k < sz; k++ ) {
            //    for ( int l = 0; l  < sz; l++ ) {
            //        slice->set(k,l, this->get(i,j,k,l));
            //    }
            //}
            int a = ijtoa(i,j);
            int stride = this->arraysize;
            double** slice = this->myarray->pointer();
            return &slice[a*stride];
        }

        SharedMatrix sliceik (int i, int k) {
            const int sz = this->arraysize;
            SharedMatrix slice = Matrix(this->arraysize,this->arraysize).clone();
            for ( int j = 0; j < sz; j++ ) {
                for ( int l = 0; l  < sz; l++ ) {
                    slice->set(i,k, this->get(i,j,k,l));
                }
            }
            return slice;
        }

        SharedMatrix slicejk (int j, int k) {
            const int sz = this->arraysize;
            SharedMatrix slice = Matrix(this->arraysize,this->arraysize).clone();
            for ( int i = 0; i < sz; i++ ) {
                for ( int l = 0; l  < sz; l++ ) {
                    slice->set(i,l, this->get(i,j,k,l));
                }
            }
            return slice;
        }

        SharedMatrix slicejl (int j, int l) {
            const int sz = this->arraysize;
            SharedMatrix slice = Matrix(this->arraysize,this->arraysize).clone();
            for ( int i = 0; i < sz; i++ ) {
                for ( int k = 0; k  < sz; k++ ) {
                    slice->set(i,k, this->get(i,j,k,l));
                }
            }
            return slice;
        }

        SharedMatrix slicekl (int k, int l) {
            const int sz = this->arraysize;
            SharedMatrix slice = Matrix(this->arraysize,this->arraysize).clone();
            for ( int i = 0; i < sz; i++ ) {
                for ( int j = 0; j  < sz; j++ ) {
                    slice->set(i,j, this->get(i,j,k,l));
                }
            }
            return slice;
        }
        inline double get (int i, int j, int k, int l) {
            //Retrieve a value from the tensor
            int a = this->ijtoa(i,j); //i*szj + j;
            int b = this->ijtoa(k,l);;//k*szl + l;
            return this->myarray->get(a,b);
        }
        inline void set (int i, int j, int k, int l, double val) {
            //Set a value in the tensor
            int a = this->ijtoa(i, j);
            int b = this->ijtoa(k,l);
            this->myarray->set(a,b,val);
        }
        int arraysize;
};

class tensor6
//class for storing and managing rank 6 tensors (!)
{
    private:
        std::unique_ptr<double***** []> data;
    public:
        int arraysize;
        tensor6(int arraysize) 
        : data{new double***** [arraysize]} 
        {
            this->arraysize = arraysize;
            for ( int i = 0; i < arraysize; i++ ) {
                this->data[i] = new double**** [arraysize];
                for ( int j = 0; j < arraysize; j++ ) {
                    this->data[i][j] = new double*** [arraysize];
                    for ( int k = 0; k < arraysize; k++ ) {
                        this->data[i][j][k] = new double** [arraysize];
                        for ( int l = 0; l < arraysize; l++ ) {
                            this->data[i][j][k][l] = new double* [arraysize];
                            for ( int m = 0; m < arraysize; m++ ) {
                                this->data[i][j][k][l][m] = new double[arraysize]; 
                                for ( int n = 0; n < arraysize; n++ ) {
                                    this->data[i][j][k][l][m][n] = 0.0;
                                }
                            }
                        }
                    }
                }
            }
        }
        ~tensor6() {
            for ( int i = 0; i < this->arraysize; i++ ) {
                for ( int j = 0; j < this->arraysize; j++ ) {
                    for ( int k = 0; k < this->arraysize; k++ ) {
                        for ( int l = 0; l < this->arraysize; l++ ) {
                            for ( int m = 0; m < this->arraysize; m++ ) {
                                delete[] this->data[i][j][k][l][m];
                            }
                            delete[] this->data[i][j][k][l];
                        }
                        delete[] this->data[i][j][k];
                    }
                    delete[] this->data[i][j];
                }
                delete[] this->data[i];
            }
        }
        double get ( int i, int j, int k, int l, int m, int n ) {
            return this->data[i][j][k][l][m][n];
        }
        void set ( int i, int j, int k, int l, int m, int n, double val ) {
            this->data[i][j][k][l][m][n] = val; 
        }
};
class phfwfn
//class to perform SCF procedure, integral transformations, and
//post HF (MP2 and CCSD) energy computations.
{
    private:
        //>Private Data

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
        void build_tijab2 (void);
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
        MintsHelper * mints;
        Vector * eval;
        SharedMatrix C;
        SharedMatrix Hcore;
        SharedMatrix HcoreSO;
        SharedMatrix FF;
        SharedMatrix Dia;
        SharedMatrix tia_new;
        tensor4 * W;
        tensor4 * tautijab;
        tensor4 * tauijab;
        tensor4 * tijab_new;
        tensor4 * Dijab;
        SharedMatrix FSO;
        SharedMatrix tia;
        tensor4 * SO_eri;
        tensor4 * MO_eri;
        tensor4 * MO_asym;
        tensor4 * AO_eri;
        tensor4 * tijab;
        tensor4 * aijab;
        int natom;
        int nbf; 
        int nmo;
        int nocc;
        int noccso;
        int maxiter;
        double ccetol;
        double E;
        double Ecorr;
        double enuc;
        // //<Public Data

        // //>Public Methods
        // void do_CCSD (void);
        void build_tijab_MP2 (void);
        double MP2viaCC (void);
        double mp2init (void);
        void do_MP2 (void);
        void do_CCSD (void);
        double mp2asym (void);
        void antisymm_MO (void);
        void build_aijab (void);

        // double do_SCF (void);
        // double compute_E (void);
        phfwfn(SharedWavefunction ref_wfn, int nbf, int nocc);
        ~phfwfn(void);
        // //<Public Methods
};
}}
#endif
