/*
 * @BEGIN LICENSE
 *
 * ddcc by Psi4 Developer, a plugin to:
 *
 * Psi4: an open-source quantum chemistry software package
 *
 * Copyright (c) 2007-2019 The Psi4 Developers.
 *
 * The copyrights for code used from other parties are included in
 * the corresponding files.
 *
 * This file is part of Psi4.
 *
 * Psi4 is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * Psi4 is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License along
 * with Psi4; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 * @END LICENSE
 */
#include <cstdlib>
#include "psi4/psi4-dec.h"
#include "psi4/psifiles.h"
#include "psi4/libdpd/dpd.h"
#include "psi4/libtrans/integraltransform.h"
#include "psi4/libpsio/psio.hpp"
#include "psi4/libmints/wavefunction.h"
#include "psi4/libpsi4util/PsiOutStream.h"
#include "psi4/liboptions/liboptions.h"
#include "psi4/libmints/molecule.h"
#include "psi4/libmints/basisset.h"
#include "psi4/libmints/matrix.h"
#include "psi4/libmints/vector.h"
#include "psi4/libmints/mintshelper.h"
#include "phf.h"



// This allows us to be lazy in getting the spaces in DPD calls
#define ID(x) ints.DPD_ID(x)

namespace psi{ 
    
namespace ddcc{

extern "C" PSI_API
int read_options(std::string name, Options &options)
{
    if (name == "DDCC" || options.read_globals()) {
        /*- The amount of information printed
            to the output file -*/
        options.add_int("PRINT", 1);
    }

    return true;
}
 

//extern "C" PSI_API
SharedMatrix build_moeri(SharedWavefunction ref_wfn, Options& options) 
{
    MintsHelper mints(MintsHelper(ref_wfn->basisset(), options, 0));
    SharedMatrix Ca = ref_wfn->Ca();
    SharedMatrix moeri = mints.mo_eri(Ca,Ca,Ca,Ca);
    return moeri;
}

extern "C" PSI_API
SharedWavefunction ddcc(SharedWavefunction ref_wfn, Options& options)
{
    int print = options.get_int("PRINT");

    // Grab the global (default) PSIO object, for file I/O
    //std::shared_ptr<PSIO> psio(_default_psio_lib_);

    // Have the reference (SCF) wavefunction, ref_wfn
    if(!ref_wfn) throw PSIEXCEPTION("SCF has not been run yet!");

    // Quickly check that there are no open shell orbitals here...
    int nirrep  = ref_wfn->nirrep();
    SharedMatrix moeri(build_moeri(ref_wfn, options));
    std::cout << ref_wfn->nalpha() + ref_wfn->nbeta()<<"\n";
    phf::phfwfn cor_wf ( ref_wfn, ref_wfn->nmo(), ref_wfn->nalpha());

    cor_wf.do_MP2();
    std::cout << std::setprecision(15) << "E[MP2] " << cor_wf.Ecorr << "\n";
    cor_wf.do_CCSD();
    
    return ref_wfn;
}

}//end ddcc
    
//>PHF WFN CONSTRUCTOR
phf::phfwfn::phfwfn(SharedWavefunction ref_wfn, int nbf, int nocc)
{
    int nmo = nbf*2;
    this->nbf =       nbf;
    this->nmo =       nmo;
    this->nocc =      nocc;
    this->noccso =    this->nocc*2;
    this->maxiter =   50;
    this->ccetol =    1E-14;
    this->eval =      ref_wfn->epsilon_a()->clone(); //mo energies
    this->mints =     new MintsHelper(ref_wfn->basisset()); //integrals
    this->C =         ref_wfn->Ca();
    this->FSO =       psi::Matrix(nmo,nmo).clone();
    this->HcoreSO =   psi::Matrix(nmo,nmo).clone();
    this->Hcore =     ref_wfn->H();//psi::Matrix(nbf,nbf).clone();
    this->MO_eri =    new phf::tensor4 (this->mints->mo_eri(
                                        this->C,this->C,this->C,this->C),
                                        this->nbf); //psi::Matrix(nmo,nmo).clone();
    this->SO_eri =    new phf::tensor4 (nmo); //spin orbital eri
    this->AO_eri =    new phf::tensor4 (nbf); //psi::Matrix(nmo,nmo).clone();
    this->FF =        psi::Matrix(nmo,nmo).clone(); //Fancy F intermediate
    this->tia =       psi::Matrix(nmo,nmo).clone();
    this->tia_new =   psi::Matrix(nmo,nmo).clone();
    this->Dia =       psi::Matrix(nmo,nmo).clone();
    this->W =         new phf::tensor4 (nmo); //W intermediate
    this->tautijab =  new phf::tensor4 (nmo);
    this->tauijab =   new phf::tensor4 (nmo);
    this->tijab =     new phf::tensor4 (nmo);
    this->tijab_new = new phf::tensor4 (nmo);
    this->Dijab =     new phf::tensor4 (nmo);
}

psi::phf::phfwfn::~phfwfn() {
}
double phf::kron ( int p, int q ) {
    return (p == q );
}
//<PHF WFN CONSTRUCTOR

//> Energy routines
double phf::phfwfn::MP2viaCC (void) {
    double tsum = 0.0;
    for ( int i = 0; i < this->noccso; i++ ) {
        for ( int j = 0; j < this->noccso; j++ ) {
            for ( int a = this->noccso; a < this->nmo; a++ ) {
                for ( int b = this->noccso; b < this->nmo; b++ ) {
                    tsum += this->SO_eri->get(i,j,a,b)*this->tijab->get(i,j,a,b);
                }
            }
        }
    }
    return tsum*0.25;
}

void phf::phfwfn::do_MP2 (void) {
    this->Ecorr = 0;
    for ( int i = 0; i < this->nocc; i++ ) {
       for ( int j = 0; j < this->nocc; j++ ) {
          for ( int a = this->nocc; a < this->nbf; a++ ) {
             for ( int b = this->nocc; b < this->nbf; b++ ) {
                this->Ecorr += this->MO_eri->get(i,a,j,b)
                             * (2*this->MO_eri->get(i,a,j,b) 
                                - this->MO_eri->get(i,b,j,a))
                             / ( this->eval->get(i)
                               + this->eval->get(j)
                               - this->eval->get(a)
                               - this->eval->get(b));
             }
          }
       }
    }
}

void phf::phfwfn::do_CCSD (void) {
    //this->print2D(this->C, 0, this->nbf, 0, this->nbf);
    this->MOtoSO();
    this->FtoSO();
    this->build_Dia();
    this->build_Dijab();
    this->build_tijab_MP2();
    std::cout << "E[MP2viaCC] " << this->MP2viaCC() << "\n";
    this->ccenergy();
    std::cout << "E[CC] " << this->Ecorr << "\n";
    double elast = 0.0;
    std::cout << "this->maxiter " << this->maxiter << "\n";
    for ( int i = 0; i < this->maxiter; i++ ) {
        std::cout << i << "\n";
        this->cciter();
        if ( std::abs(this->Ecorr - elast) < this->ccetol ) {
            break;
        }
        elast = this->Ecorr;
    }
}
void phf::phfwfn::cciter (void) {
    this->build_tautijab(); //Eqn 9
    this->build_tauijab(); //Eqn10
    this->build_Fae(); //Eqn 3
    this->build_Fme(); //Eqn 5
    this->build_Fmi(); //Eqn 4
    this->build_Wmnij(); //Eqn 6
    this->build_Wmbej(); //Eqn 8
    this->build_Wabef(); //Eqn 7
    this->build_tia(); //Eqn 1
    this->build_tijab(); //Eqn 2
    this->tiacpy();
    this->tijabcpy();
    this->ccenergy();
    std::cout << "E[CC] " << this->Ecorr << "\n";
}
void phf::phfwfn::ccenergy (void) {
    double Ecc = 0.0;
    for ( int i = 0; i < this->noccso; i++ ) {
        for ( int a = this->noccso; a < this->nmo; a++ ) { 
            Ecc +=  this->FSO->get(i,a)
                  * this->tia->get(i,a);
        }
    }
    for ( int i = 0; i < this->noccso; i++ ) {
        for ( int j = 0; j < this->noccso; j++ ) {
            for ( int a = this->noccso; a < this->nmo; a++ ) {
                for ( int b = this->noccso; b < this->nmo; b++ ) {
                    Ecc += 0.25*this->SO_eri->get(i,j,a,b)
                               *this->tijab->get(i,j,a,b);
                    Ecc += 0.5*this->SO_eri->get(i,j,a,b)
                              *this->tia->get(i,a)
                              *this->tia->get(j,b);           
                }
            }
        }
    }
    this->Ecorr = Ecc;
}
//< Energy routines

//> Builds (F, W, etc)
//
void phf::phfwfn::build_Wabef (void) {
    //Equation 7 from Stanton90
    //with a large virtual space, I think this will be the chonkiest term.
    double tsum1 = 0.0;
    double tsum2 = 0.0;
    for ( int a = this->noccso; a < this->nmo; a++ ) {
        for ( int b = this->noccso; b < this->nmo; b++ ) {
            for ( int e = this->noccso; e < this->nmo; e++ ) {
                for ( int f = this->noccso; f < this->nmo; f++ ) {
                //Summation 1: over m (occ)
                tsum1 = 0.0;
                for ( int m = 0; m < this->noccso; m++ ) {
                    tsum1 += this->tia->get( m, b )
                           * this->SO_eri->get( a, m, e, f);
                    tsum1 -= this->tia->get( m, a )
                           * this->SO_eri->get( b, m, e, f);
                }

                //Summation 2: over m, n (occ, occ)
                tsum2 = 0.0;
                for ( int m = 0; m < this->noccso; m++ ) {
                    for ( int n = 0; n < this->noccso; n++ ) {
                        tsum2 += this->tauijab->get ( m, n, a, b)
                               * this->SO_eri->get( m, n , e, f);
                    }
                }
                tsum2 *= 0.25;

                this->W->set( a, b, e, f, this->SO_eri->get( a, b, e, f)
                                             - tsum1 + tsum2);
                }
            }
        }
    }
}

void phf::phfwfn::build_Wmnij (void) {
    double tsum1 = 0.0;
    double tsum2 = 0.0;
    for ( int m = 0; m < this->noccso; m++ ) {
        for ( int n = 0; n < this->noccso; n++ ) {
            for ( int i = 0; i < this->noccso; i++ ) {
                for ( int j = 0; j < this->noccso; j++ ) {
                    //Summation 1: over e (vir)
                    tsum1 = 0.0;
                    for ( int e = this->noccso; e < this->nmo; e++ ) {
                        //p(ij) - p(ji)
                        tsum1 += this->tia->get( j, e )
                              * this->SO_eri->get( m, n, i, e);
                        tsum1 -= this->tia->get( i, e )
                               * this->SO_eri->get( m, n, j, e);
                    }

                    //Summation 2: over e, f (vir, vir)
                    tsum2 = 0.0;
                    for ( int e = this->noccso; e < this->nmo; e++ ) {
                        for ( int f = this->noccso; f < this->nmo; f++ ) {
                            tsum2 += this->tauijab->get( i, j, e, f)
                                   * this->SO_eri->get( m, n, e, f);
                        }
                    }
                    tsum2 *= 0.25;

                    this->W->set( m, n, i, j, (this->SO_eri->get( m, n, i, j)
                                             + tsum1 + tsum2));
                }
            }
        }
    }
}
void phf::phfwfn::build_Wmbej (void) {
    //Equation 8 from Stanton90
    double tsum1 = 0.0;
    double tsum2 = 0.0;
    double tsum3 = 0.0;
    for ( int m = 0; m < this->noccso; m++ ) {
        for ( int b = this->noccso; b < this->nmo; b++ ) { 
            for ( int e = this->noccso; e < this->nmo; e++ ) {
                for ( int j = 0; j < this->noccso; j++ ) {
                    //Summation 1: over f (vir) 
                    tsum1 = 0.0;
                    for ( int f = this->noccso; f < this->nmo; f++ ) {
                        tsum1 += this->tia->get( j, f )
                              * this->SO_eri->get( m, b, e, f);
                    }

                    //Summation 2: over n (occ)
                    tsum2 = 0.0;
                    for ( int n = 0; n < this->noccso; n++ ) {
                        tsum2 += this->tia->get( n, b )
                               * this->SO_eri->get( m, n, e, j);
                    }

                    //Summation 3: over n, f ( occ , vir )
                    tsum3 = 0.0;
                    for ( int n = 0; n < this->noccso; n++ ) {
                        for ( int f = this->noccso; f < this->nmo; f++ ) {
                            tsum3 += (0.5*this->tijab->get( j, n, f, b)
                                      + this->tia->get( j, f)
                                      * this->tia->get( n, b)
                                     )
                                   * this->SO_eri->get( m, n, e, f);
                        }
                    }
                    this->W->set( m, b, e, j,  (this->SO_eri->get( m, b, e, j)
                                                  + tsum1 - tsum2 - tsum3));
                }
            }
        }
    }
}

void phf::phfwfn::build_Fae (void) {
    //Eqn 3 from Stanton90
    double tsum1 = 0;
    double tsum2 = 0;
    double tsum3 = 0;
    for ( int a = this->noccso; a < this->nmo; a++ ) {
        for ( int e = this->noccso; e < this->nmo; e++ ) {
            //std::cout << a << " " << e << "\n";
            //First summation: over m (occ)
            tsum1 = 0.0;
            for ( int m = 0; m < this->noccso; m++ ) {
                tsum1 +=  this->FSO->get( m, e )
                        * this->tia->get( m, a );
            }
            tsum1 *= 0.5;
            //Second summation: over m, f (occ, vir)
            tsum2 = 0.0;
            for ( int m = 0; m < this->noccso; m++ ) {
                for ( int f = this->noccso; f < this->nmo; f++ ) {
                    tsum2 +=  this->tia->get( m, f)
                            * this->SO_eri->get( m, a, f, e);
                }
            }
            ////Third summation: over m, n, f (occ, occ, vir)
            tsum3 = 0.0;
            for ( int m = 0; m < this->noccso; m++ ) {
                for ( int n = 0; n < this->noccso; n++ ) {
                    for ( int f = this->noccso; f < this->nmo; f++ ) {
                        tsum3 +=  this->tautijab->get( m, n, a, f)
                                * this->SO_eri->get( m, n, e, f);
                    }
                }
            }
            tsum3 *= 0.5;
            this->FF->set( a, e,
                             ((1.0 - psi::phf::kron(a,e))*this->FSO->get( a, e)
                              - tsum1 + tsum2 - tsum3)
                           ); 

        }
    }
}   
void phf::phfwfn::build_Fmi (void) {
    //Equation 4 from Stanton90
    double tsum1 = 0.0;
    double tsum2 = 0.0;
    double tsum3 = 0.0;
    for ( int m = 0; m < this->noccso; m++ ) {
        for ( int i = 0; i < this->noccso; i++ ) {
            //First summation: over e (vir)
            tsum1 = 0.0;
            for ( int e = this->noccso; e < this->nmo; e++ ) {
                tsum1 +=  this->tia->get( i, e )
                        * this->FSO->get( m, e );
            }
            tsum1 *= 0.5;

            //Second summation: over n, e (occ, vir)
            tsum2 = 0.0;
            for ( int n = 0; n < this->noccso; n++ ) {
                for ( int e = this->noccso; e < this->nmo; e++ ) {
                    tsum2 +=  this->tia->get( n, e )
                            * this->SO_eri->get(m,n,i,e);
                }
            }

            //Third summation: over n, e , f (occ, vir, vir)
            tsum3 = 0.0;
            for ( int n = 0; n < this->noccso; n++ ) {
                for ( int e = this->noccso; e < this->nmo; e++ ) {
                    for ( int f = this->noccso; f < this->nmo; f++ ) {
                        tsum3 +=  this->tautijab->get( i, n, e, f)
                                * this->SO_eri->get( m, n, e, f);
                    }
                }
            }
            tsum3 *= 0.5;
            this->FF->set( m, i,
                        ((1.0 - kron(m,i))*this->FSO->get( m, i )
                        + tsum1 + tsum2 + tsum3 ));
        }
    }
}

void phf::phfwfn::build_Fme (void) {
    //Equation 5 in Stanton90
    double tsum = 0.0;
    for ( int m = 0; m < this->noccso; m++ ) {
        for ( int e = this->noccso; e < this->nmo; e++ ) {
            //Summation: over n, f (occ, vir)
            tsum = 0.0;
            for ( int n = 0; n < this->noccso; n++ ) {
                for ( int f = this->noccso; f < this->nmo; f++ ) {
                    tsum +=  this->tia->get( n, f)
                            * this->SO_eri->get( m, n, e, f);
                }
            }
            this->FF->set( m, e,
                          (this->FSO->get( m, e )
                           + tsum ));
        }
    }
}

void phf::phfwfn::build_tautijab (void) {
    //Eqn 9 from Stanton90
    for ( int i = 0; i < this->noccso; i++ ) {
        for ( int j = 0; j < this->noccso; j++ ) {
            for ( int a = this->noccso; a < this->nmo; a++ ) {
                for ( int b = this->noccso; b < this->nmo; b++ ) {
                    this->tautijab->set( i, j, a, b,
                                               (this->tijab->get( i, j, a, b)
                                               + 0.5 *
                                               (
                                                  this->tia->get( i, a)
                                                * this->tia->get( j, b)
                                                - this->tia->get( i, b)
                                                * this->tia->get( j, a)
                                               )));

                }
            }
        }
    }
}

void phf::phfwfn::build_tauijab (void) {
    //Eqn 10 from Stanton90
    for ( int i = 0; i < this->noccso; i++ ) {
        for ( int j = 0; j < this->noccso; j++ ) {
            for ( int a = this->noccso; a < this->nmo; a++ ) {
                for ( int b = this->noccso; b < this->nmo; b++ ) {
                    this->tauijab->set(i,j,a,b,
                                             (this->tijab->get( i, j, a, b)
                                             + (
                                                  this->tia->get( i, a )
                                                * this->tia->get( j, b )
                                                - this->tia->get( i, b )
                                                * this->tia->get( j, a )
                                               )));
                }
            }
        }
    }
}

void phf::phfwfn::build_Dia (void) {
    for ( int i = 0; i < this->nmo; i++ ) {
        for ( int a = 0; a < this->nmo; a++ ) {
            this->Dia->set ( i, a,
                            this->FSO->get( i, i )
                            - this->FSO->get( a, a));
        }
    }
}

void phf::phfwfn::build_Dijab (void) {
    for ( int i = 0; i < this->nmo; i++ ){
        for ( int j = 0; j < this->nmo; j++ ) {
            for ( int a = 0; a < this->nmo; a++ ) {
                for ( int b = 0; b < this->nmo; b++) {
                    this->Dijab->set(i,j,a,b,
                        this->FSO->get( i, i)
                      + this->FSO->get( j, j)
                      - this->FSO->get( a, a)
                      - this->FSO->get( b, b));
                }
            }
        }
    }
}

void phf::phfwfn::build_tia (void) {
    //Equation 1 from Stanton90
    double tiatmp = 0.0; //will be assigned via gsl_matrix_set to this->tia
    double tsum1 = 0.0;
    double tsum2 = 0.0;
    double tsum3 = 0.0;
    double tsum4 = 0.0;
    double tsum5 = 0.0;
    double tsum6 = 0.0;
    for ( int i = 0; i < this->noccso; i++ ) {
        for ( int a = this->noccso; a < this->nmo; a++ ) {

            //Summation 1: over e ( vir )
            tsum1 = 0.0;
            for ( int e = this->noccso; e < this->nmo; e++ ) {
                tsum1 +=  this->tia->get( i, e )
                        * this->FF->get( a, e );
            }

            //Summation 2: over m ( occ )
            tsum2 = 0.0;
            for ( int m = 0; m < this->noccso; m++ ) {
                tsum2 +=  this->tia->get( m, a )
                        * this->FF->get( m, i );
            }

            //Summation 3: over m, e ( occ, vir )
            tsum3 = 0.0;
            for ( int m = 0; m < this->noccso; m++ ) {
                for ( int e = this->noccso; e < this->nmo; e++ ) {
                    tsum3 +=  this->tijab->get(i,m,a,e)
                            * this->FF->get( m, e );
                }
            }

            //Summation 4: over n, f ( occ, vir )
            tsum4 = 0.0;
            for ( int n = 0; n < this->noccso; n++ ) {
                for ( int f = this->noccso; f < this->nmo; f++ ) {
                    tsum4 +=  this->tia->get( n, f )
                            * this->SO_eri->get(n,a,i,f);
                }
            }

            //Summation 5: over m, e, f ( occ, vir, vir )
            tsum5 = 0.0;
            for ( int m = 0; m < this->noccso; m++ ) {
                for ( int e = this->noccso; e < this->nmo; e++ ) {
                    for ( int f = this->noccso; f < this->nmo; f++ ) {
                        tsum5 += this->tijab->get(i,m,e,f)
                               * this->SO_eri->get(m,a,e,f);
                    }
                }
            }
            tsum5 *= 0.5;

            //Summation 6: over m, e , n ( occ, vir, occ )
            tsum6 = 0.0;
            for ( int m = 0; m < this->noccso; m++ ) {
                for ( int e = this->noccso; e < this->nmo; e++ ) {
                    for ( int n = 0; n < this->noccso; n++ ) {
                        tsum6 += this->tijab->get(m,n,a,e)
                               * this->SO_eri->get(n,m,e,i);
                    }
                }
            }
            tsum6 *= 0.5;

            tiatmp = this->FSO->get ( i , a )
                   + tsum1 - tsum2 + tsum3 - tsum4 - tsum5 - tsum6;
            tiatmp /= this->Dia->get( i, a );
            this->tia_new->set( i, a, tiatmp );
        }
    }
}
void phf::phfwfn::tiacpy (void) {
    for ( int i =0; i < this->noccso; i++ ) {
        for ( int a = this->noccso; a < this->nmo; a++ ) {
            this->tia->set( i , a , this->tia_new->get( i , a ));
        }
    }
}

void phf::phfwfn::build_tijab (void) {
    double tijabtmp = 0.0;
    double tsum1 = 0.0;
    double tsum1a = 0.0;
    double tsum2 = 0.0;
    double tsum2a = 0.0;
    double tsum3 = 0.0;
    double tsum4 = 0.0;
    double tsum5 = 0.0;
    double tsum6 = 0.0;
    double tsum7 = 0.0;
    for ( int i = 0; i < this->noccso; i++ ) {
        for ( int j = 0; j < this->noccso; j++ ) {
            for ( int a = this->noccso; a < this->nmo; a++ ) {
                for ( int b = this->noccso; b < this->nmo; b++ ) {
                    tijabtmp = this->SO_eri->get( i, j, a, b);
                    //term #2
                    //<<--PERMUTATION P_(ab)-->>//
                    //P_(ab) = p(ab) - p(ba)
                    //p(ab) part
                    //Summation 1: over e ( vir )
                    tsum1 = 0.0;
                    for ( int e = this->noccso; e < this->nmo; e++ ) {
                        //Sub-Summation 1a: over m ( occ )
                        tsum1a = 0.0;
                        for ( int m = 0; m < this->noccso; m++ ) {
                            tsum1a += this->tia->get( m, b )
                                  * this->FF->get( m, e );
                        }
                        tsum1a *= 0.5;
                        tsum1 += this->tijab->get( i, j, a, e)
                               * ( this->FF->get( b, e )
                                 - tsum1a
                                 );
                    }
                    //at this point tsum1 = p(ab) part
                    //p(ba) part
                    //Summation 1': over e ( vir )
                    for ( int e = this->noccso; e < this->nmo; e++ ) {
                        //Sub-Summation 1a': over m ( occ )
                        tsum1a = 0.0;
                        for ( int m = 0; m < this->noccso; m++ ) {
                            tsum1a += this->tia->get( m, a )
                                    * this->FF->get( m, e );
                        }
                        tsum1a *= 0.5;
                        // -= here ...
                        tsum1 -= this->tijab->get( i, j, b, e)
                              * ( this->FF->get( a, e )
                                - tsum1a
                                );
                    }
                    // ... means that at this point tsum1 = p(ab) - p(ba)

                    //-->>PERMUTATION P_(ab)<<--//

                    //term #3
                    //<<--PERMUTATION P_(ij)-->>//
                    //P_(ij) = p(ij) - p(ji)
                    //p(ij) part
                    //Summation 2: over m ( occ )
                    tsum2 = 0.0;
                    for ( int m = 0; m < this->noccso; m++ ) {
                        //Sub-Summation 2a: over e ( vir )
                        tsum2a = 0.0;
                        for ( int e = this->noccso; e < this->nmo; e++ ) {
                            tsum2a += this->tia->get( j, e )
                                  * this->FF->get( m, e );
                        }
                        tsum2a *= 0.5;
                        tsum2 += this->tijab->get( i, m, a, b)
                               * ( this->FF->get( m, j )
                                 + tsum2a
                                 );
                    }
                    //at this point tsum2 = p(ij) part
                    //p(ji) part
                    //Summation 2': over m ( occ )
                    for ( int m = 0; m < this->noccso; m++ ) {
                        //Sub-Summation 2a': over e ( vir )
                        tsum2a = 0.0;
                        for ( int e = this->noccso; e < this->nmo; e++ ) {
                            tsum2a += this->tia->get( i, e )
                                    * this->FF->get( m, e );
                        }
                        tsum2a *= 0.5;
                        // -= here ...
                        tsum2 -= this->tijab->get( j, m, a, b)
                              * ( this->FF->get( m, i )
                                + tsum2a
                                );
                    }
                    // ... means that at this point tsum2 = p(ij) - p(ji)

                    //-->>PERMUTATION P_(ab)<<--//
                    //term #4
                    //Summation : over m, n ( occ, occ )
                    tsum3 = 0.0;
                    for ( int m = 0; m < this->noccso; m++ ) {
                        for ( int n = 0; n < this->noccso; n++ ) {
                            tsum3 += this->tauijab->get( m, n, a, b)
                                   * this->W->get( m, n, i, j);
                        }
                    }
                    tsum3 *= 0.5;

                    //term #5
                    //Summation : over e, f ( vir, vir )
                    tsum4 = 0.0;
                    for ( int e = this->noccso; e < this->nmo; e++ ) {
                        for ( int f = this->noccso; f < this->nmo; f++ ) {
                            tsum4 += this->tauijab->get( i, j, e, f)
                                   * this->W->get( a, b, e, f);
                        }
                    }
                    tsum4 *= 0.5;
                    //term #6
                    //<<--PERMUTATION P_(ij)P_(ab)-->>//
                    //this is a nested permutation
                    //P_(ij)P_(ab)[f] = P_(ij)[p(ab)f - p(ba)f]
                    // = p(ij)[p(ab)f - p(ba)f] - p(ji)[p(ab)f - p(ba)f]
                    // = p(ij)p(ab)f - p(ij)p(ba)f - p(ji)p(ab)f + p(ji)p(ba)f

                    //Summation : over m, e ( occ , vir )
                    tsum5 = 0.0;
                    for ( int m = 0; m < this->noccso; m++ ) {
                        for ( int e = this->noccso; e < this->nmo; e++ ) {
                            //p(ij)p(ab) part
                            tsum5 += this->tijab->get( i, m, a, e)
                                  * this->W->get( m, b, e, j);
                            tsum5 -= this->tia->get( i, e )
                                   * this->tia->get( m, a )
                                   * this->SO_eri->get( m, b, e, j);

                            //p(ij)p(ba)f part
                            tsum5 -= this->tijab->get( i, m, b, e)
                                  * this->W->get( m, a, e, j);
                            tsum5 += this->tia->get( i, e )
                                  * this->tia->get( m, b )
                                  * this->SO_eri->get( m, a, e, j);

                            //p(ji)p(ab)f part
                            tsum5 -= this->tijab->get( j, m, a, e)
                                  * this->W->get( m, b, e, i);
                            tsum5 += this->tia->get( j, e )
                                  * this->tia->get( m, a )
                                  * this->SO_eri->get( m, b, e, i);

                            //p(ji)p(ba)f part
                            tsum5 += this->tijab->get( j, m, b, e)
                                  * this->W->get( m, a, e, i);
                            tsum5 -= this->tia->get( j, e )
                                  * this->tia->get( m, b )
                                  * this->SO_eri->get( m, a, e, i);
                        }
                    }
                    //-->>PERMUTATION P_(ij)P_(ab)<<--//
                    //term #7
                    //<<--PERMUTATION P_(ij)-->>//
                    tsum6 = 0.0;
                    //Summation : over e (vir)
                    for ( int e = this->noccso; e < this->nmo; e++ ) {
                        //p(ij) part
                        tsum6 += this->tia->get( i, e )
                               * this->SO_eri->get( a, b, e, j);

                        //p(ji) part
                        tsum6 -= this->tia->get( j, e )
                               * this->SO_eri->get( a, b, e, i);
                    }
                    //-->>PERMUTATION P_(ij)<<--//

                    //term #8
                    //<<--PERMUTATION P_(ab)-->>//
                    //Summation : over m ( occ )
                    tsum7 = 0.0;
                    for ( int m = 0; m < this->noccso; m++ ) {
                        //p(ab) part
                        tsum7 += this->tia->get( m, a )
                               * this->SO_eri->get( m, b, i, j);

                        //p(ba) part
                        tsum7 -= this->tia->get( m, b )
                               * this->SO_eri->get( m, a, i, j);
                    }
                    tijabtmp += tsum1;
                    tijabtmp -= tsum2;
                    tijabtmp += tsum3;
                    tijabtmp += tsum4;
                    tijabtmp += tsum5;
                    //std::cout << "TIJAB " << tijabtmp << "\n";
                    tijabtmp += tsum6;
                    tijabtmp -= tsum7;
                    tijabtmp /= this->Dijab->get( i, j, a, b);
                    this->tijab_new->set( i, j, a, b, tijabtmp);
                }
            }
        }
    }
}

void phf::phfwfn::tijabcpy (void) {
    for ( int i =0; i < this->noccso; i++ ) {
        for ( int j =0; j < this->noccso; j++ ) {
            for ( int a = this->noccso; a < this->nmo; a++ ) {
                for ( int b = this->noccso; b < this->nmo; b++ ) {
                    this->tijab->set( i, j, a, b, this->tijab_new->get( i, j, a, b));
                }
            }
        }
    }
}
void phf::phfwfn::build_tijab_MP2 (void) {
    for ( int i = 0; i < this->noccso; i++ ) {
        for ( int j = 0; j < this->noccso; j++ ) {
            for ( int a = this->noccso; a < this->nmo; a++ ) {
                for ( int b =  this->noccso; b < this->nmo; b++ ) {
                    this->tijab->set (i,j,a,b,
                        this->SO_eri->get(i,j,a,b)
                        / (this->Dijab->get(i,j,a,b)));
                }
            }
        }
    } 
}

//> Transformations
void phf::phfwfn::FtoSO (void) {
    //Translates a phfwfn 's Fock matrix from MO -> SO basis (>>SPIN ORBITAL<<)
    double tsum = 0.0;
    this->HcoretoSO();
    for ( int p = 0; p < this->nmo; p++ ) {
        for ( int q = 0; q < this->nmo; q++ ) {
            tsum = 0.0;
            tsum += this->HcoreSO->get(p,q);
            for ( int m = 0; m < this->noccso; m++ ) {
                tsum += this->SO_eri->get(p,m,q,m);
            }
            this->FSO->set(p,q,tsum);
        }
    }
}

void phf::phfwfn::HcoretoSO (void) {
    //Translates a phfwfn 's Hcore matrix from MO -> SO basis (>>SPIN ORBITAL<<)
    double spinint;
    int pp,qq;
    for ( int p = 0; p < this->nmo; p++ ) {
        for ( int q = 0; q < this->nmo; q++ ) {
            spinint = 0.0;
            pp = p/2;
            qq = q/2;    
            for ( int mu = 0; mu < this->nbf; mu++ ) {
                for ( int nu = 0; nu < this->nbf; nu++ ) {
                    spinint += this->C->get (mu, qq)
                             * this->C->get (nu, pp)
                             * this->Hcore->get (mu, nu); 
                }
            }
            spinint *= (p%2 == q%2);
            this->HcoreSO->set (p, q, spinint);
        }
    } 
}

void phf::phfwfn::MOtoSO (void) {
    //Translates a phfwfn 's MO eri -> SO (>>SPIN ORBITAL<< not XXsymm orbitalXX)
    std::cout << "in MOtoSO ... \n";
    int pp = 0;
    int qq = 0;
    int rr = 0;
    int ss = 0;
    double spinint1;
    double spinint2;
    for ( int p = 0; p < this->nmo; p++ ) {
        for ( int q = 0; q < this->nmo; q++ ) {
            for ( int r = 0; r < this->nmo; r++ ) {
                for ( int s = 0; s < this->nmo; s++ ) {
                    pp = p/2;
                    qq = q/2;
                    rr = r/2;
                    ss = s/2;
                    spinint1 = this->MO_eri->get(pp,qq,rr,ss) * ( p%2 == q%2 ) * (r%2 == s%2);
                    spinint2 = this->MO_eri->get(pp,ss,qq,rr) * ( p%2 == s%2 ) * (q%2 == r%2);
                    this->SO_eri->set(p,r,q,s, spinint1 - spinint2);
                }
            }
        }
    }
}

void phf::phfwfn::print2D (SharedMatrix A, int lb1, int ub1, int lb2, int ub2) {
    std::cout << "in prin2D\n";
    for ( int i = lb1; i < ub1; i++ ) {
        for ( int j = lb2; j < ub2; j++ ) {
            std::cout << std::setprecision(15) << A->get(i, j) << " ";        
        }
        std::cout << "\n"; 
    }
}

phf::tensor4::tensor4(SharedMatrix inmat, int parraysize) : myarray{ inmat->clone() }
{
    this->arraysize = parraysize;
}
phf::tensor4::tensor4(int parraysize)
    : myarray{ psi::Matrix(parraysize*parraysize,parraysize*parraysize).clone() }
{
    this->arraysize = parraysize;
}
} //end psi
