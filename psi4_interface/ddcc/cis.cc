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
#include "ci.h"

namespace psi {
namespace cis {
extern "C" PSI_API
int read_options(std::string name, Options &options)
{
    if (name == "CIS" || options.read_globals()) {
        /*- The amount of information printed
            to the output file -*/
        options.add_int("PRINT", 1);
    }

    return true;
}
extern "C" PSI_API 
SharedWavefunction cis (SharedWavefunction ref_wfn, Options& options)
{
    int print = options.get_int("PRINT");
    // Have the reference (SCF) wavefunction, ref_wfn
    if(!ref_wfn) throw PSIEXCEPTION("SCF has not been run yet!");
    phf::phfwfn corwf ( ref_wfn, ref_wfn->nmo(), ref_wfn->nalpha());
    int nocc = corwf.noccso;
    int nbf = corwf.nmo;
    int N = nocc * ( nbf - nocc );
    SharedMatrix H ( Matrix(N,N).clone() );
    int i = 0;
    int a = 10;
    std::cout << Hsingle ( i, i, a, a, &corwf ) << "\n";
    return ref_wfn;
}
}//end cis
}//end psi
