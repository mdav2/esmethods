#include "determinant.h"
#include "psi4/libmints/matrix.h"
#include "psi4/libmints/vector.h"
#include "psi4/libmints/mintshelper.h"
namespace psi {
namespace cis {

//double correction_d ( phfwfn * corwf );
void test (void);
double Hsingle ( int i, int j, int a, int b, psi::phf::phfwfn * corwf);

}//end phf
}//end psi
