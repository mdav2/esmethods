#include <cstdlib>
#include <memory>
#include <omp.h>
#include "phf.h"
#include "determinant.h"
#include "psi4/psi4-dec.h"
#include "psi4/libmints/matrix.h"
#include "psi4/libmints/vector.h"
#include "psi4/libmints/mintshelper.h"

namespace psi {
namespace phf {
phfwfn::phfwfn(SharedWavefunction ref_wfn, int nbf, int nocc)
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
    this->MO_eri =    new tensor4 (this->mints->mo_eri(
                                        this->C,this->C,this->C,this->C),
                                        this->nbf); //psi::Matrix(nmo,nmo).clone();
    this->SO_eri =    new phf::tensor4 (nmo); //spin orbital eri
    this->MO_asym =   new tensor4 (nbf); //psi::Matrix(nmo,nmo).clone();
    this->AO_eri =    new tensor4 (nbf); //psi::Matrix(nmo,nmo).clone();
    this->FF =        Matrix(nmo,nmo).clone(); //Fancy F intermediate
    this->tia =       Matrix(nmo,nmo).clone();
    this->tia_new =   Matrix(nmo,nmo).clone();
    this->Dia =       Matrix(nmo,nmo).clone();
    this->W =         new tensor4 (nmo); //W intermediate
    this->tautijab =  new tensor4 (nmo);
    this->tauijab =   new tensor4 (nmo);
    this->tijab =     new tensor4 (nmo);
    this->tijab_new = new tensor4 (nmo);
    this->Dijab =     new tensor4 (nmo);
    this->aijab =     new tensor4 (nbf);
    this->MOtoSO();
    this->FtoSO();
}
phfwfn::~phfwfn() {
    delete this->SO_eri;
    delete this->AO_eri;
    delete this->W;
    delete this->tautijab;
    delete this->tauijab;
    delete this->tijab;
    delete this->tijab_new;
    delete this->Dijab;
    delete this->mints;
}
double kron ( int p, int q ) {
    return (p == q );
}

//>ENERGY ROUTINES
double phfwfn::mp2init (void) {

    this->build_Dijab();
    this->build_tijab_MP2();
    return this->MP2viaCC();
}
double phfwfn::MP2viaCC (void) {
    double tsum = 0.0;
    int low = 0;
    int high = this->noccso;
    #pragma omp parallel for default(shared) reduction(+:tsum)
    for ( int i = low; i < high; ++i ) {
        for ( int j = 0; j < this->noccso; j++ ) {
            for ( int a = this->noccso; a < this->nmo; a++ ) {
                for ( int b = this->noccso; b < this->nmo; b++ ) {
                    tsum += this->SO_eri->get(i,j,a,b)*this->tijab->get(i,j,a,b);
                }
            }
        }
    }
    //} // END PRAGMA OMP PARALLEL
    return tsum*0.25;
}
void phf::phfwfn::do_CCSD (void) {
    //this->print2D(this->C, 0, this->nbf, 0, this->nbf);
 
    this->MOtoSO();
    this->FtoSO();
    this->build_Dia();
    this->build_Dijab();
    this->build_tijab_MP2();
    psi::outfile->Printf("E[MP2   ] %20.14f\n",this->MP2viaCC());
    this->ccenergy();
    psi::outfile->Printf("E[CC;%3d] %20.14f\n", 0, this->Ecorr);
    double elast = 0.0;
    std::cout << "this->maxiter " << this->maxiter << "\n";
    for ( int i = 0; i < this->maxiter; i++ ) {
        //std::cout << i << "\n";
        this->cciter();
        psi::outfile->Printf("E[CC;%3d] %20.14f\n",i+1,this->Ecorr); 
        if ( std::abs(this->Ecorr - elast) < this->ccetol ) {
            //psi::outfile->Printf("-"*40);
            psi::outfile->Printf("\nE[CC;final] %20.14f\n",this->Ecorr); 
            break;
        }
        elast = this->Ecorr;
    }
}
void phf::phfwfn::cciter (void) {
    this->build_tautijab(); //Eqn 9
    this->build_tauijab(); //Eqn10
    #pragma omp parallel sections
    {
        #pragma omp section
        {
        this->build_Fae(); //Eqn 3
        this->build_Fme(); //Eqn 5
        this->build_Fmi(); //Eqn 4
        }
        #pragma omp section
        {
        this->build_Wmnij(); //Eqn 6
        }
        #pragma omp section
        {
        this->build_Wmbej(); //Eqn 8
        }
        #pragma omp section
        {
        this->build_Wabef(); //Eqn 7
        }

    }

    this->build_tia(); //Eqn 1
    this->build_tijab(); //Eqn 2
    #pragma omp parallel sections
    {
    #pragma omp section
    {
    this->tiacpy();
    }
    #pragma omp section
    {
    this->tijabcpy();
    }
    }
    this->ccenergy();
}
void phf::phfwfn::ccenergy (void) {
    double Ecc = 0.0;
    #pragma omp parallel sections
    {
    #pragma omp section
    {
    for ( int i = 0; i < this->noccso; i++ ) {
        for ( int a = this->noccso; a < this->nmo; a++ ) { 
            Ecc +=  this->FSO->get(i,a)
                  * this->tia->get(i,a);
        }
    }
    }
    #pragma omp section
    {
    #pragma omp parallel for default(shared) collapse(4) reduction(+:Ecc)
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
    }
    }
    this->Ecorr = Ecc;
}
//< Energy routines
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
//> Builds (F, W, etc)
//
void phf::phfwfn::build_Wabef (void) {
    //Equation 7 from Stanton90
    //with a large virtual space, I think this will be the chonkiest term.
    #pragma omp parallel for default(shared) collapse(4)
    for ( int a = this->noccso; a < this->nmo; a++ ) {
        for ( int b = this->noccso; b < this->nmo; b++ ) {
            for ( int e = this->noccso; e < this->nmo; e++ ) {
                for ( int f = this->noccso; f < this->nmo; f++ ) {
                double tsum1 = 0.0;
                double tsum2 = 0.0;
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
    #pragma omp parallel for default(shared) collapse(4)
    for ( int m = 0; m < this->noccso; m++ ) {
        for ( int n = 0; n < this->noccso; n++ ) {
            for ( int i = 0; i < this->noccso; i++ ) {
                for ( int j = 0; j < this->noccso; j++ ) {
                    double tsum1 = 0.0;
                    double tsum2 = 0.0;
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
    #pragma omp parallel for default(shared) collapse(4)
    for ( int m = 0; m < this->noccso; m++ ) {
        for ( int b = this->noccso; b < this->nmo; b++ ) { 
            for ( int e = this->noccso; e < this->nmo; e++ ) {
                for ( int j = 0; j < this->noccso; j++ ) {
                    //Summation 1: over f (vir) 
                    double tsum1 = 0.0;
                    double tsum2 = 0.0;
                    double tsum3 = 0.0;
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
            tsum2 = 0.0;
            tsum3 = 0.0;
            for ( int m = 0; m < this->noccso; m++ ) {
                tsum1 +=  this->FSO->get( m, e )
                        * this->tia->get( m, a );
                for ( int f = this->noccso; f < this->nmo; f++ ) {
                    tsum2 +=  this->tia->get( m, f)
                            * this->SO_eri->get( m, a, f, e);
                }
                for ( int n = 0; n < this->noccso; n++ ) {
                    for ( int f = this->noccso; f < this->nmo; f++ ) {
                        tsum3 +=  this->tautijab->get( m, n, a, f)
                                * this->SO_eri->get( m, n, e, f);
                    }
                }
            }
            tsum1 *= 0.5;
            //Second summation: over m, f (occ, vir)
            ////Third summation: over m, n, f (occ, occ, vir)
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
            }
            tsum1 *= 0.5;

            //Second summation: over n, e (occ, vir)
            tsum2 = 0.0;
            tsum3 = 0.0;
            for ( int e = this->noccso; e < this->nmo; e++ ) {
                tsum1 +=  this->tia->get( i, e )
                        * this->FSO->get( m, e );
                for ( int n = 0; n < this->noccso; n++ ) {
                    tsum2 +=  this->tia->get( n, e )
                            * this->SO_eri->get(m,n,i,e);
                    for ( int f = this->noccso; f < this->nmo; f++ ) {
                        tsum3 +=  this->tautijab->get( i, n, e, f)
                                * this->SO_eri->get( m, n, e, f);
                    }
                }
            }

            //Third summation: over n, e , f (occ, vir, vir)
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
    #pragma omp parallel for default(shared) collapse(2)
    for ( int i = 0; i < this->noccso; i++ ) {
        for ( int a = this->noccso; a < this->nmo; a++ ) {
            double tiatmp = 0.0; //will be assigned via gsl_matrix_set to this->tia
            double tsum1 = 0.0;
            double tsum2 = 0.0;
            double tsum3 = 0.0;
            double tsum4 = 0.0;
            double tsum5 = 0.0;
            double tsum6 = 0.0;

            //Summation 1: over e ( vir )
            tsum1 = 0.0;
            for ( int e = this->noccso; e < this->nmo; e++ ) {
                tsum1 +=  this->tia->get( i, e )
                        * this->FF->get( a, e );
            }

            //Summation 2: over m ( occ )

            //Summation 3: over m, e ( occ, vir )
            tsum3 = 0.0;
            for ( int m = 0; m < this->noccso; m++ ) {
                tsum2 +=  this->tia->get( m, a )
                        * this->FF->get( m, i );
                for ( int e = this->noccso; e < this->nmo; e++ ) {
                    tsum3 +=  this->tijab->get(i,m,a,e)
                            * this->FF->get( m, e );
                    for ( int f = this->noccso; f < this->nmo; f++ ) {
                        tsum5 += this->tijab->get(i,m,e,f)
                               * this->SO_eri->get(m,a,e,f);
                    }
                    for ( int n = 0; n < this->noccso; n++ ) {
                        tsum6 += this->tijab->get(m,n,a,e)
                               * this->SO_eri->get(n,m,e,i);
                    }
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

            tsum5 *= 0.5;
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
    #pragma omp parallel for default(shared) collapse(4) schedule(dynamic)
    for ( int i = 0; i < this->noccso; i++ ) {
        for ( int j = 0; j < this->noccso; j++ ) {
            for ( int a = this->noccso; a < this->nmo; a++ ) {
                for ( int b = this->noccso; b < this->nmo; b++ ) {
                    double tijabtmp = 0.0;
                    double tsum1 = 0.0;
                    double tsum1a = 0.0;
                    double tsum1b = 0.0;
                    double tsum2 = 0.0;
                    double tsum2a = 0.0;
                    double tsum3 = 0.0;
                    double tsum4 = 0.0;
                    double tsum5 = 0.0;
                    double tsum6 = 0.0;
                    double tsum7 = 0.0;
                    tijabtmp = this->SO_eri->get( i, j, a, b);
                    //term #2
                    //<<--PERMUTATION P_(ab)-->>//
                    //P_(ab) = p(ab) - p(ba)
                    //p(ab) part
                    //Summation 1: over e ( vir )
                    for ( int e = this->noccso; e < this->nmo; e++ ) {
                        //Sub-Summation 1a: over m ( occ )
                        tsum1a = 0.0;
                        tsum1b = 0.0;
                        for ( int m = 0; m < this->noccso; m++ ) {
                            tsum1a += this->tia->get( m, b )
                                  * this->FF->get( m, e );
                            tsum1b += this->tia->get( m, a )
                                    * this->FF->get( m, e );
                        }
                        tsum1a *= 0.5;
                        tsum1b *= 0.5;
                        tsum1 += this->tijab->get( i, j, a, e)
                               * ( this->FF->get( b, e )
                                 - tsum1a
                                 );
                        // -= here ...
                        tsum1 -= this->tijab->get( i, j, b, e)
                              * ( this->FF->get( a, e )
                                - tsum1b
                                );
                    }

                    for ( int m = 0; m < this->noccso; m++ ) {
                        //p(ab) part
                        tsum7 += this->tia->get( m, a )
                               * this->SO_eri->get( m, b, i, j);

                        //p(ba) part
                        tsum7 -= this->tia->get( m, b )
                               * this->SO_eri->get( m, a, i, j);
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
                    //}

                    //at this point tsum2 = p(ij) part
                    //p(ji) part
                    //Summation 2': over m ( occ )
                    //for ( int m = 0; m < this->noccso; m++ ) {
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
                        for ( int n = 0; n < this->noccso; n++ ) {
                            tsum3 += this->tauijab->get( m, n, a, b)
                                   * this->W->get( m, n, i, j);
                        }
                    }
                    // ... means that at this point tsum2 = p(ij) - p(ji)

                    //-->>PERMUTATION P_(ab)<<--//
                    //term #4
                    //Summation : over m, n ( occ, occ )
                    tsum3 *= 0.5;

                    //term #5
                    //Summation : over e, f ( vir, vir )
                    for ( int e = this->noccso; e < this->nmo; e++ ) {
                        //p(ij) part
                        tsum6 += this->tia->get( i, e )
                               * this->SO_eri->get( a, b, e, j);

                        //p(ji) part
                        tsum6 -= this->tia->get( j, e )
                               * this->SO_eri->get( a, b, e, i);
                        for ( int f = this->noccso; f < this->nmo; f++ ) {
                            tsum4 += this->tauijab->get( i, j, e, f)
                                   * this->W->get( a, b, e, f);
                        }
                    }
                    tsum4 *= 0.5;
                    
                    //term #7
                    //<<--PERMUTATION P_(ij)-->>//
                    //Summation : over e (vir)
                    //-->>PERMUTATION P_(ij)<<--//

                    //term #8
                    //<<--PERMUTATION P_(ab)-->>//
                    //Summation : over m ( occ )
                    //term #6
                    //<<--PERMUTATION P_(ij)P_(ab)-->>//
                    //this is a nested permutation
                    //P_(ij)P_(ab)[f] = P_(ij)[p(ab)f - p(ba)f]
                    // = p(ij)[p(ab)f - p(ba)f] - p(ji)[p(ab)f - p(ba)f]
                    // = p(ij)p(ab)f - p(ij)p(ba)f - p(ji)p(ab)f + p(ji)p(ba)f

                    //Summation : over m, e ( occ , vir )
                    tsum5 = 0.0;
                    #pragma omp parallel for default(shared) reduction(+:tsum5) schedule(dynamic) collapse(2)
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
    #pragma omp parallel for default(shared) schedule(dynamic) collapse(4)
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

double phf::phfwfn::mp2asym(void) {
    double e = 0.0;
    for ( int i = 0 ; i < this->nocc; i++ ) {
        for ( int j = 0; j < this->nocc; j++ ) {
            for ( int a = this->nocc; a < this->nbf; a++ ) {
                for ( int b = this->nocc; b < this->nbf; b++ ) {
                    //e += this->MO_eri->get(i,a,j,b)*(2*this->MO_eri->get(i,a,j,b)
                    //                                 - this->MO_eri->get(i,b,j,a))/
                    //                    (this->eval->get(i)
                    //                     + this->eval->get(j)
                    //                     - this->eval->get(a)
                    //                     - this->eval->get(b));      
                    e += 0.25*this->aijab->get(i,j,a,b)*this->MO_asym->get(i,j,a,b);
                }
            }
        }
    }
    return e;
}

void phf::phfwfn::build_aijab(void) {
    for ( int i = 0 ; i < this->nocc; i++ ) {
        for ( int j = 0; j < this->nocc; j++ ) {
            for ( int a = this->nocc; a < this->nbf; a++ ) {
                for ( int b = this->nocc; b < this->nbf; b++ ) {
                    this->aijab->set (i,j,a,b,this->MO_asym->get(a,b,i,j)/(
                                  this->eval->get(i) + this->eval->get(j)
                                - this->eval->get(a) - this->eval->get(b)));
                }
            }
        }
    }
}

void phf::phfwfn::antisymm_MO (void) {
    double int1 = 0.0;
    double int2 = 0.0;
    for ( int p = 0; p < this->nbf; p++ ) {
        for ( int q = 0; q < this->nbf; q++ ) {
            for ( int r = 0; r < this->nbf; r++ ) {
                for ( int s = 0; s < this->nbf; s++ ) {
                    int1 = this->MO_eri->get(p,q,r,s);
                    int2 = this->MO_eri->get(p,s,q,r);
                    this->MO_asym->set(p,r,q,s, int1 - int2);
                }
            }
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
phf::tensor4::~tensor4() {
}
}//END PHF
}//END PSI
