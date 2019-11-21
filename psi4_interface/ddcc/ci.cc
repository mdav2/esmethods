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

}//end ph
}//end psi
