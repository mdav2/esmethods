#include <cstdlib>
#include <memory>
#include <omp.h>
#include "phf.h"
#include "determinant.h"
#include "psi4/psi4-dec.h"
#include "psi4/libmints/matrix.h"
#include "psi4/libmints/vector.h"
#include "psi4/libmints/mintshelper.h"
#include "psi4/libmints/wavefunction.h"
#include "xtensor/xarray.hpp"
#include "xtensor/xtensor.hpp"
#include "xtensor/xio.hpp"

namespace psi {
namespace phf {
double kron ( int p, int q ) {
    return (p == q );
}
int ijtoa ( int i, int j , int sz ) {
    return i*sz + j;
}
void make_phfwfn (phfwfn& pwf, SharedWavefunction rwf)
{
    pwf.mints = new MintsHelper(rwf->basisset());
    pwf.nalpha = rwf->nalpha(); 
    pwf.nbeta = rwf->nbeta();
    pwf.nmo = rwf->nmo();
    pwf.nvira = pwf.nmo - pwf.nalpha;
    pwf.nvirb = pwf.nmo - pwf.nbeta;

    xt::xtensor<double,2>::shape_type shape_f = {pwf.nmo,pwf.nmo}; 
    xt::xtensor<double,4>::shape_type shape_ijab_oovv = {pwf.nalpha,pwf.nalpha,pwf.nvira,pwf.nvira};
    xt::xtensor<double,4>::shape_type shape_iJaB_oovv = {pwf.nalpha,pwf.nbeta ,pwf.nvira,pwf.nvirb};
    xt::xtensor<double,4>::shape_type shape_iJAb_oovv = {pwf.nalpha,pwf.nbeta ,pwf.nvirb,pwf.nvira};
    xt::xtensor<double,4>::shape_type shape_IJAB_oovv = {pwf.nbeta ,pwf.nbeta ,pwf.nvirb,pwf.nvirb};
    xt::xtensor<double,4>::shape_type shape_IjAb_oovv = {pwf.nbeta ,pwf.nalpha,pwf.nvirb,pwf.nvira};
    xt::xtensor<double,4>::shape_type shape_IjaB_oovv = {pwf.nbeta ,pwf.nalpha,pwf.nvira,pwf.nvirb};
    xt::xtensor<double,4>::shape_type shape_pqrs = {pwf.nmo, pwf.nmo, pwf.nmo, pwf.nmo};

    pwf.ijab = xt::zeros<double>(shape_pqrs);
    pwf.iJaB = xt::zeros<double>(shape_pqrs);
    pwf.iJAb = xt::zeros<double>(shape_pqrs);
    pwf.IJAB = xt::zeros<double>(shape_pqrs);
    pwf.IjAb = xt::zeros<double>(shape_pqrs);
    pwf.IjaB = xt::zeros<double>(shape_pqrs);

    pwf.tijab = xt::zeros<double>(shape_ijab_oovv);
    pwf.tiJaB = xt::zeros<double>(shape_iJaB_oovv);
    pwf.tiJAb = xt::zeros<double>(shape_iJAb_oovv);
    pwf.tIJAB = xt::zeros<double>(shape_IJAB_oovv);
    pwf.tIjAb = xt::zeros<double>(shape_IjAb_oovv);
    pwf.tIjaB = xt::zeros<double>(shape_IjaB_oovv);

    pwf.Dijab = xt::zeros<double>(shape_ijab_oovv);
    pwf.DiJaB = xt::zeros<double>(shape_iJaB_oovv);
    pwf.DiJAb = xt::zeros<double>(shape_iJAb_oovv);
    pwf.DIJAB = xt::zeros<double>(shape_IJAB_oovv);
    pwf.DIjAb = xt::zeros<double>(shape_IjAb_oovv);
    pwf.DIjaB = xt::zeros<double>(shape_IjaB_oovv);

    pwf.Ca = xt::zeros<double>(shape_f);
    pwf.Cb = xt::zeros<double>(shape_f);
    pwf.fa = xt::zeros<double>(shape_f);
    pwf.fb = xt::zeros<double>(shape_f);

    SharedMatrix ca = rwf->Ca();
    SharedMatrix cb = rwf->Cb();
    SharedMatrix mo_ijab = pwf.mints->mo_eri(ca, ca, ca, ca);
    SharedMatrix mo_iJaB = pwf.mints->mo_eri(ca, cb, ca, cb);
    SharedMatrix mo_iJAb = pwf.mints->mo_eri(ca, cb, cb, ca);
    SharedMatrix mo_IJAB = pwf.mints->mo_eri(cb, cb, cb, cb);
    SharedMatrix mo_IjAb = pwf.mints->mo_eri(cb, ca, cb, ca);
    SharedMatrix mo_IjaB = pwf.mints->mo_eri(cb, ca, ca, cb);
    make_C (pwf,rwf->Ca(),rwf->Cb());
    make_f (pwf,rwf->Fa(),rwf->Fb());
    make_ijabs (pwf, mo_ijab, mo_iJaB, mo_iJAb, mo_IJAB, mo_IjAb, mo_IjaB);
    make_Dijabs (pwf);
    make_tijabs (pwf)
}

void make_C (phfwfn& pwf, SharedMatrix ca, SharedMatrix cb) {
    for ( int i = 0; i < pwf.nmo; i++ ) {
        for ( int j = 0; j < pwf.nmo; j++ ) {
            pwf.Ca[{i,j}] = ca->get(i,j); 
            pwf.Cb[{i,j}] = cb->get(i,j); 
        }
    }
}

void make_f (phfwfn& pwf, SharedMatrix fa, SharedMatrix fb) {
    for ( int i = 0; i < pwf.nmo; i++ ) {
        for ( int j = 0; j < pwf.nmo; j++ ) {
            pwf.fa[{i,j}] = fa->get ( i, j );
            pwf.fb[{i,j}] = fb->get ( i, j ); 
        }
    }
}

void make_ijabs (phfwfn& pwf, SharedMatrix mo_ijab, SharedMatrix mo_iJaB,
    //fills in spin cases for ERI tensors
    //eqns 4 - 10 in cookbook
                              SharedMatrix mo_iJAb, SharedMatrix mo_IJAB, 
                              SharedMatrix mo_IjAb, SharedMatrix mo_IjaB ) {
    for ( int i = 0; i < pwf.nmo; i++ ) {
        for ( int j = 0; j < pwf.nmo; j++ ) {
            for ( int a = 0; a < pwf.nmo; a++ ) {
                for ( int b = 0; b < pwf.nmo; b++ ) {
                    int xa = ijtoa(i,a,pwf.nmo);
                    int xb = ijtoa(j,b,pwf.nmo);
                    int xaa = ijtoa(i,b,pwf.nmo);
                    int xbb = ijtoa(j,a,pwf.nmo);
                    pwf.ijab[{i,j,a,b}] = mo_ijab->get(xa,xb) - mo_ijab->get(xaa,xbb);
                    pwf.iJaB[{i,j,a,b}] = mo_iJaB->get(xa,xb);
                    pwf.iJAb[{i,j,a,b}] = -1*mo_iJAb->get(xaa,xbb);
                    pwf.IJAB[{i,j,a,b}] = mo_IJAB->get(xa,xb) - mo_IJAB->get(xaa,xbb);
                    pwf.IjAb[{i,j,a,b}] = mo_IjAb->get(xa,xb);
                    pwf.IjaB[{i,j,a,b}] = -1*mo_IjaB->get(xaa,xbb);
                }
            }
        }
    }
}

void make_Dijabs (phfwfn& pwf) {
    //fills in Dijab spin cases
    //case 1 : Dijab : eqn 16
    for ( int i = 0; i < pwf.nalpha; i++ ) {
        for ( int j = 0; j < pwf.nalpha; j++ ) {
            for ( int a = pwf.nalpha; a < pwf.nmo; a++ ) {
                for ( int b = pwf.nalpha, b < pwf.nmo; b++ ) {
                    pwf.Dijab[{i,j,a,b}] = pwf.fa[{i,i}] + pwf.fa[{j,j}]    
                                         - pwf.fa[{a,b}] - pwf.fa[{a,b}]; 
                }
            }
        }
    }
    //case 2 : DiJaB : eqn 18
    for ( int i = 0; i < pwf.nalpha; i++ ) {
        for ( int j = 0; j < pwf.nbeta; j++ ) {
            for ( int a = pwf.nalpha; a < pwf.nmo; a++ ) {
                for ( int b = pwf.nbeta, b < pwf.nmo; b++ ) {
                    pwf.DiJaB[{i,j,a,b}] = pwf.fa[{i,i}] + pwf.fb[{j,j}]    
                                         - pwf.fa[{a,b}] - pwf.fb[{a,b}]; 
                }
            }
        }
    }
    //case 3 : DiJAb : eqn 20
    for ( int i = 0; i < pwf.nalpha; i++ ) {
        for ( int j = 0; j < pwf.nbeta; j++ ) {
            for ( int a = pwf.nbeta; a < pwf.nmo; a++ ) {
                for ( int b = pwf.nalpha, b < pwf.nmo; b++ ) {
                    pwf.DiJAb[{i,j,a,b}] = pwf.fa[{i,i}] + pwf.fb[{j,j}]    
                                         - pwf.fb[{a,b}] - pwf.fa[{a,b}]; 
                }
            }
        }
    }
    //case 4 : DIJAB : eqn 22
    for ( int i = 0; i < pwf.nbeta; i++ ) {
        for ( int j = 0; j < pwf.nbeta; j++ ) {
            for ( int a = pwf.nbeta; a < pwf.nmo; a++ ) {
                for ( int b = pwf.nbeta, b < pwf.nmo; b++ ) {
                    pwf.DIJAB[{i,j,a,b}] = pwf.fb[{i,i}] + pwf.fb[{j,j}]    
                                         - pwf.fb[{a,b}] - pwf.fb[{a,b}]; 
                }
            }
        }
    }
    //case 5 : DIjAb : eqn 24
    for ( int i = 0; i < pwf.nbeta; i++ ) {
        for ( int j = 0; j < pwf.nalpha; j++ ) {
            for ( int a = pwf.nbeta; a < pwf.nmo; a++ ) {
                for ( int b = pwf.nalpha, b < pwf.nmo; b++ ) {
                    pwf.DIjAb[{i,j,a,b}] = pwf.fb[{i,i}] + pwf.fa[{j,j}]    
                                         - pwf.fb[{a,b}] - pwf.fa[{a,b}]; 
                }
            }
        }
    }
    //case 1 : DIjaB : eqn 26
    for ( int i = 0; i < pwf.nbeta; i++ ) {
        for ( int j = 0; j < pwf.nalpha; j++ ) {
            for ( int a = pwf.nalpha; a < pwf.nmo; a++ ) {
                for ( int b = pwf.nbeta, b < pwf.nmo; b++ ) {
                    pwf.DIjaB[{i,j,a,b}] = pwf.fb[{i,i}] + pwf.fa[{j,j}]    
                                         - pwf.fa[{a,b}] - pwf.fb[{a,b}]; 
                }
            }
        }
    }
}
}
}
}
