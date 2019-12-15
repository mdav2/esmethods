#include "phf.h"
#include "psi4/libmints/matrix.h"
#include "psi4/libmints/vector.h"
#include "psi4/libmints/mintshelper.h"
namespace psi {
namespace phf {
void print_hello_world (void);
double pert_triples (phfwfn * corwf);
double full_triples (phfwfn * inwf);
}
}//end psi
