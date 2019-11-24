#include "determinant.h"
#include "psi4/libmints/matrix.h"
#include "psi4/libmints/vector.h"
#include "psi4/libmints/mintshelper.h"
namespace psi {
namespace cis {

//double correction_d ( phfwfn * corwf );
void test (void);
double Hsingle ( int i, int j, int a, int b, psi::phf::phfwfn * corwf);
void build_uijab (phf::tensor4 * uijab, phf::tensor4 * SO_eri, SharedMatrix bb, phf::phfwfn * corwf ) ;
SharedMatrix build_nu ( phf::tensor4 * SO_eri, SharedMatrix bb, phf::tensor4 * tijab,
                        phf::phfwfn * corwf );
double w_cis_d ( phf::tensor4 * uijab, phf::tensor4 * Dijab, double w, SharedMatrix bb, 
                 SharedMatrix nu, phf::phfwfn * corwf ) ;

}//end phf
}//end psi
