#include "phf.h"
#include "ci.h"
#include "psi4/libmints/matrix.h"
#include "psi4/libmints/vector.h"
#include "psi4/libmints/mintshelper.h"
namespace psi {
namespace cis {

void test (void) {
    std::cout << "VOID\n";
}

double Hsingle ( int i, int j, int a, int b, phf::phfwfn * corwf) {
    return   phf::kron(i,j)*corwf->FSO->get(a,b) 
           - phf::kron(a,b)*corwf->FSO->get(i,j)
           + corwf->SO_eri->get(a,j,i,b);
}

void build_uijab (phf::tensor4 * uijab, phf::tensor4 * MO_eri, SharedMatrix bb, phf::phfwfn * corwf) {
    for ( int i = 0; i < corwf->nocc; i++ ) {
        for ( int j = 0; j < corwf->nocc; j++ ) {
            for ( int a = corwf->nocc; a < corwf->nbf; a++ ) {
                for ( int b = corwf->nocc; b < corwf->nbf; b++ ) {
                    double temp = 0.0;
                    for ( int c = corwf->nocc; c < corwf->nbf; c++ ) {
                        temp +=     MO_eri->get(a,b,c,j)*bb->get(i,c) 
                                -   MO_eri->get(a,b,c,i)*bb->get(j,c);
                    }          
                    for ( int k = 0; k < corwf->nocc; k++ ) {
                        temp +=     MO_eri->get(k,a,i,j)*bb->get(k,b)
                                -   MO_eri->get(k,b,i,j)*bb->get(k,a);
                    }
                    uijab->set( i, j, a, b, temp);
                }
            }
        }
    } 
}

double w_cis_d ( phf::tensor4 * uijab, phf::tensor4 * Dijab, double w, SharedMatrix bb, 
                 SharedMatrix nu, phf::phfwfn * corwf ) {
    double wcisd = 0.0;
    double temp = 0.0;
    double dijab = 0.0;
    std::cout << w << "\n";
    for ( int i = 0; i < corwf->nocc; i++ ) {
        for ( int j = 0; j < corwf->nocc; j++ ) {
            for ( int a = corwf->nocc; a < corwf->nbf; a++ ) {
                for ( int b = corwf->nocc; b < corwf->nbf; b++ ) {
                    dijab  = (corwf->eval->get(i) + corwf->eval->get(j) - corwf->eval->get(a)
                              - corwf->eval->get(b));
                    temp = -1*uijab->get(i,j,a,b)*uijab->get(i,j,a,b)/(dijab - w);
                    if ( temp > 1E-10 ) {
                        //std::cout << temp << "\n";
                    }
                    wcisd += temp;
                }
            }
        }
    }
    wcisd *= 0.25;
    for ( int i = 0; i < corwf->noccso; i++ ) {
        for ( int a = corwf->noccso; a < corwf->nmo; a++ ) {
            wcisd += bb->get(i,a)*nu->get(i,a);
        }
    }
    return wcisd;
}
SharedMatrix build_nu (phf::tensor4 * MO_eri, SharedMatrix bb, phf::tensor4 * tijab, phf::phfwfn * corwf) {
//eqn 13 of Head-Gordon94
    SharedMatrix nu ( bb->clone() );
    for ( int  i = 0; i < corwf->nocc; i++ ) {
        for ( int  a = corwf->nocc; a < corwf->nbf; a++ ) {
            double temp = 0.0;
            for ( int  j = 0; j < corwf->nocc; j++ ) {
                for ( int k = 0; k < corwf->nocc; k++ ) {
                    for ( int b = corwf->nocc; b < corwf->nbf; b++ ) {
                        for ( int c = corwf->nocc; c < corwf->nbf; c++ ) {
                            temp +=   MO_eri->get (j,k,b,c)
                                    * ( bb->get(i,b)*tijab->get(2*j,2*k,2*c,2*a)
                                      + bb->get(j,a)*tijab->get(2*i,2*k,2*c,2*b)
                                      + 2*bb->get(j,b)*tijab->get(2*j,2*k,2*a,2*c));
                        }
                    }
                }
            }
            temp *= 0.5;
            nu->set ( i, a, temp);
        }
    }
    return nu;
}

}//end ph
}//end psi
