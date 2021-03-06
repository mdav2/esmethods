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
        options.add_bool("PERTD",false);
    }

    return true;
}
extern "C" PSI_API 
SharedWavefunction cis (SharedWavefunction ref_wfn, Options& options)
{
    psi::outfile->Printf("-------------------------------------------\n");
    psi::outfile->Printf("<<-- Configuration Interaction Singles -->>\n");
    psi::outfile->Printf("-------------------------------------------\n\n");
    psi::outfile->Printf(" ( spin-unadapted ) \n");
    psi::outfile->Printf(" using full diagonalization\n");
    psi::outfile->Printf(" using in-core algorithm\n");
    int print = options.get_int("PRINT");
    int pertd = options.get_bool("PERTD");
    if ( pertd == 1 ) {
        psi::outfile->Printf(" will do CIS(D) correction\n\n");
    }
    // Have the reference (SCF) wavefunction, ref_wfn
    if(!ref_wfn) throw PSIEXCEPTION("SCF has not been run yet!");
    phf::phfwfn corwf ( ref_wfn, ref_wfn->nmo(), ref_wfn->nalpha());

    int nocc = corwf.noccso;
    int nbf = corwf.nmo;
    int nvir = nbf - nocc;
    int N = nocc * ( nvir );
    SharedMatrix H ( Matrix(N,N).clone() );
    SharedMatrix vecs ( Matrix(N,N).clone() );
    Vector vals(N);
    int x;
    int y;
    psi::outfile->Printf("building Hamiltonian matrix ... "); 
    for ( int i = 0; i < nocc; i++ ) {
        for ( int j = 0; j < nocc; j++ ) {
            for ( int a = nocc; a < nbf; a++ ) {
                #pragma omp parallel for shared(H,corwf)
                for ( int b = nocc; b < nbf; b++ ) {
                    H->set( i*( nvir ) + (a - nocc), j*( nvir ) + (b - nocc), Hsingle ( i, j, a, b, &corwf ));   
                }
            }
        }
    }
    psi::outfile->Printf("done!\n"); 
    psi::outfile->Printf("diagonalize Hamiltonian ... ");
    H->diagonalize(vecs,vals);//could replace with a call to davidson
    psi::outfile->Printf("done!\n");
    psi::outfile->Printf(" Excitation energies (a.u.)\n");
    psi::outfile->Printf(" ------------------\n");
    for ( int i = 0; i < N; i++ ) {
        psi::outfile->Printf("%20.14f\n",vals.get(i));
    }
    psi::outfile->Printf(" ------------------\n");
    if ( pertd == 1 ) {
        psi::outfile->Printf(" <<---------------------------------->>\n");
        psi::outfile->Printf("                CIS(D)\n");
        psi::outfile->Printf(" perturbative doubles correction to CIS\n");
        psi::outfile->Printf(" <<---------------------------------->>\n");

        psi::outfile->Printf(" Initializing MP2 T2 amplitudes ...");
        //double mp2e = corwf.mp2init();
        //corwf.do_MP2();
        corwf.antisymm_MO();
        corwf.build_aijab();
        std::cout << "EMP2 " << corwf.mp2asym() << "\n";
        psi::outfile->Printf(" Done! \n");
        psi::outfile->Printf("MP2 energy %20.14f\n",corwf.Ecorr);
        corwf.do_MP2();
        std::cout << "EM2 " << corwf.Ecorr << "\n";
        corwf.do_CCSD() ; 
        std::cout << "EM2 " << corwf.Ecorr << "\n";
        psi::outfile->Printf(" Building nu matrix ...");
        SharedMatrix nu = build_nu (corwf.MO_asym, vecs, corwf.tijab, &corwf);    
        psi::outfile->Printf(" Done! \n");
        psi::outfile->Printf(" Building u matrix ...");
        phf::tensor4 uijab ( corwf.nbf );
        build_uijab ( &uijab, corwf.MO_asym, vecs, &corwf);
        psi::outfile->Printf(" Done! \n");
        std::cout << w_cis_d ( &uijab, corwf.Dijab, vals.get(0), vecs, nu, &corwf) << "\n";
        
    }
    psi::outfile->Printf("-----------------------------------------\n");
    psi::outfile->Printf("-->>Configuration Interaction Singles<<--\n");
    psi::outfile->Printf("-----------------------------------------\n");
    return ref_wfn;
}
}//end cis
}//end psi
