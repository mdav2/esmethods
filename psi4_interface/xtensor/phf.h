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
#include "xtensor/xarray.hpp"
#include "xtensor/xtensor.hpp"
/*
Data structures and functions for computing post-SCF energies.
Reference is arbitrary, and all routines are coded for spin-orbitals.
General approach:
    1) PSI reference wfn -> xtensor shapes
        a) Ca
        b) Cb
        ...
    2) PSI data structures -> phfwfn struct
    3) Energies
note:
1) and 2) are mostly implemented.
3) is not. 
TODO:
T2 guess amplitudes
MP2 energy (checkpoint for accuracy)
CC equations
*/

namespace psi{ namespace phf {
double kron (int p, int q); 
        // //<Public Methods
struct phfwfn
{
    long unsigned int nmo;
    long unsigned int nalpha;
    long unsigned int nbeta;
    long unsigned int nvira;
    long unsigned int nvirb;
    int maxiter;
    //int ccetol;
    MintsHelper * mints;

    //fock
    xt::xtensor<double,2> fa;
    xt::xtensor<double,2> fb;

    //coeff
    xt::xtensor<double,2> Ca;
    xt::xtensor<double,2> Cb;

    //ERI spin cases
    xt::xtensor<double,4> ijab;
    xt::xtensor<double,4> iJaB;
    xt::xtensor<double,4> iJAb;
    xt::xtensor<double,4> IJAB;
    xt::xtensor<double,4> IjAb;
    xt::xtensor<double,4> IjaB;

    //T2 spin cases
    xt::xtensor<double,4> tijab;
    xt::xtensor<double,4> tiJaB;
    xt::xtensor<double,4> tiJAb;
    xt::xtensor<double,4> tIJAB;
    xt::xtensor<double,4> tIjAb;
    xt::xtensor<double,4> tIjaB;

    //Dijab spin cases
    xt::xtensor<double,4> Dijab;
    xt::xtensor<double,4> DiJaB;
    xt::xtensor<double,4> DiJAb;
    xt::xtensor<double,4> DIJAB;
    xt::xtensor<double,4> DIjAb;
    xt::xtensor<double,4> DIjaB;
};

//the functions below should be called in roughly this order
void make_phfwfn (phfwfn&, SharedWavefunction);
void make_C (phfwfn&, SharedMatrix, SharedMatrix);
void make_f (phfwfn&, SharedMatrix, SharedMatrix);
void make_ijabs (phfwfn&, SharedMatrix, SharedMatrix, SharedMatrix,
                          SharedMatrix, SharedMatrix, SharedMatrix);
void make_Dijabs (phfwfn&);
void make_tijabs_MP2 (phfwfn&);
}}
#endif
