#include "triples.h"
//#include <omp.h>
#include <iostream>
#include "psi4/libmints/matrix.h"
#include "psi4/libmints/vector.h"
#include "psi4/libmints/mintshelper.h"


namespace psi {
namespace phf {

double pert_triples (phfwfn * corwf) {
//do_pert triples
    int nmo = corwf->nmo;
    int noccso = corwf->noccso;
    double Ept = 0.0;
    //tensor6 Dijkabc (nmo);
    //tensor6 tijkabc_c (nmo);
    //tensor6 tijkabc_d (nmo);

    //build Dijkabc denominator array
    
    #pragma omp parallel for collapse(4) default(shared) reduction(+:Ept)
    for ( int i = 0; i < noccso; i++ ) {
        for ( int j = 0; j < noccso; j++ ) {
            for ( int k = 0; k < noccso; k++ ) {
                for ( int a = noccso; a < nmo; a++ ) {
                    double Dijkabc = 0.0;
                    double tijkabc_c = 0.0;
                    double tijkabc_d = 0.0;
                    double temp = 0.0;
                    for ( int b = noccso; b < nmo; b++ ) {
                        for ( int c = noccso; c < nmo; c++ ) {
                            tijkabc_c = 0.0;
                            tijkabc_d = 0.0;
                            Dijkabc =      corwf->FSO->get( i, i)
                                         + corwf->FSO->get( j, j)
                                         + corwf->FSO->get( k, k)
                                         - corwf->FSO->get( a, a)
                                         - corwf->FSO->get( b, b)
                                         - corwf->FSO->get( c, c);
                            //P(i/jk)P(a/bc)f(abcijk)  
                            // = P(i/jk)[f(abcijk) - f(bacijk) - f(cbaijk)] 
                            // = (f(abcijk) - f(bacijk) - f(cbaijk) 
                            //   - (f(abcjik) - f(bacjik) - f(cbajik))
                            //   - (f(abckji) - f(backji) - f(cbakji))
                            //
                            // Final Form:
                            // = + f(abcijk) - f(bacijk) - f(cbaijk)
                            //   - f(abcjik) + f(bacjik) + f(cbajik)
                            //   - f(abckji) + f(backji) + f(cbakji)
                            temp = 0.0;
                            temp += corwf->tia->get(i,a)*corwf->SO_eri->get(j,k,b,c);
                            temp -= corwf->tia->get(i,b)*corwf->SO_eri->get(j,k,a,c);
                            temp -= corwf->tia->get(i,c)*corwf->SO_eri->get(j,k,b,a);

                            temp -= corwf->tia->get(j,a)*corwf->SO_eri->get(i,k,b,c);
                            temp += corwf->tia->get(j,b)*corwf->SO_eri->get(i,k,a,c);
                            temp += corwf->tia->get(j,c)*corwf->SO_eri->get(i,k,b,a);

                            temp -= corwf->tia->get(k,a)*corwf->SO_eri->get(j,i,b,c);
                            temp += corwf->tia->get(k,b)*corwf->SO_eri->get(j,i,a,c);
                            temp += corwf->tia->get(k,c)*corwf->SO_eri->get(j,i,b,a);

                            temp /= Dijkabc;
                            tijkabc_d = temp; //0->temp);

                            //P(i/jk)P(a/bc)f(abcijk)  
                            // = P(i/jk)[f(abcijk) - f(bacijk) - f(cbaijk)] 
                            // = (f(abcijk) - f(bacijk) - f(cbaijk) 
                            //   - (f(abcjik) - f(bacjik) - f(cbajik))
                            //   - (f(abckji) - f(backji) - f(cbakji))
                            //
                            // Final Form:
                            // = + f(abcijk) - f(bacijk) - f(cbaijk)
                            //   - f(abcjik) + f(bacjik) + f(cbajik)
                            //   - f(abckji) + f(backji) + f(cbakji)
                            temp = 0.0;
                            #pragma omp parallel for default(shared) reduction(+:temp)
                            for ( int e = corwf->noccso; e < corwf->nmo; e++ ) {
                               temp +=   corwf->tijab->get(j,k,a,e)
                                       * corwf->SO_eri->get(e,i,b,c);
                               temp -=   corwf->tijab->get(j,k,b,e)
                                       * corwf->SO_eri->get(e,i,a,c);
                               temp -=   corwf->tijab->get(j,k,c,e)
                                       * corwf->SO_eri->get(e,i,b,a);
                               temp -=   corwf->tijab->get(i,k,a,e)
                                       * corwf->SO_eri->get(e,j,b,c);
                               temp +=   corwf->tijab->get(i,k,b,e)
                                       * corwf->SO_eri->get(e,j,a,c);
                               temp +=   corwf->tijab->get(i,k,c,e)
                                       * corwf->SO_eri->get(e,j,b,a);
                               temp -=   corwf->tijab->get(j,i,a,e)
                                       * corwf->SO_eri->get(e,k,b,c);
                               temp +=   corwf->tijab->get(j,i,b,e)
                                       * corwf->SO_eri->get(e,k,a,c);
                               temp +=   corwf->tijab->get(j,i,c,e)
                                            * corwf->SO_eri->get(e,k,b,a);
                            }
                            #pragma omp parallel for default(shared) reduction(+:temp)
                            for ( int m = 0; m < corwf->noccso; m++ ) {
                                temp -=   corwf->tijab->get(i,m,b,c)
                                        * corwf->SO_eri->get(m,a,j,k);
                                temp +=   corwf->tijab->get(i,m,a,c)
                                        * corwf->SO_eri->get(m,b,j,k);
                                temp +=   corwf->tijab->get(i,m,b,a)
                                        * corwf->SO_eri->get(m,c,j,k);
                                temp +=   corwf->tijab->get(j,m,b,c)
                                        * corwf->SO_eri->get(m,a,i,k);
                                temp -=   corwf->tijab->get(j,m,a,c)
                                        * corwf->SO_eri->get(m,b,i,k);
                                temp -=   corwf->tijab->get(j,m,b,a)
                                        * corwf->SO_eri->get(m,c,i,k);
                                temp +=   corwf->tijab->get(k,m,b,c)
                                        * corwf->SO_eri->get(m,a,j,i);
                                temp -=   corwf->tijab->get(k,m,a,c)
                                        * corwf->SO_eri->get(m,b,j,i);
                                temp -=   corwf->tijab->get(k,m,b,a)
                                        * corwf->SO_eri->get(m,c,j,i);
                            }
                            temp /= Dijkabc;
                            tijkabc_c = temp;

                            #pragma omp critical
                            Ept += (1.0/36.0)*tijkabc_c*Dijkabc*(
                                                        tijkabc_c + tijkabc_d);
                
                        }
                    }
                }
            }
        }
    }  
    return Ept;
}

double full_triples (phfwfn * corwf) {
    return 0.0;
}

}//end triples
}//end psi
