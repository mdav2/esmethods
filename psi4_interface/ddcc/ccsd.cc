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
#include "triples.h"



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
        options.add_bool("DO_TRIPLES",false);
    }

    return true;
}

extern "C" PSI_API
SharedWavefunction ccsd (SharedWavefunction ref_wfn, Options& options)
{
    int print = options.get_int("PRINT");
    // Have the reference (SCF) wavefunction, ref_wfn
    if(!ref_wfn) throw PSIEXCEPTION("SCF has not been run yet!");

    // Quickly check that there are no open shell orbitals here...
    int nirrep  = ref_wfn->nirrep();
    phf::phfwfn cor_wf ( ref_wfn, ref_wfn->nmo(), ref_wfn->nalpha());

    //cor_wf.do_MP2();
    //std::cout << std::setprecision(15) << "E[MP2] " << cor_wf.Ecorr << "\n";
    psi::outfile->Printf("--------------\n");
    psi::outfile->Printf("<<-- CCSD -->>\n");
    psi::outfile->Printf("--------------\n\n");
    int doTriples = options.get_bool("DO_TRIPLES");
    std::cout << doTriples << "\n";
    psi::outfile->Printf("DO_TRIPLES %d\n",doTriples);
    cor_wf.do_CCSD();
    if ( doTriples == 1 ) {
        psi::outfile->Printf("E[(T)] %20.14f\n",phf::pert_triples(&cor_wf));
    }
    return ref_wfn;
}

}//end ddcc
} //end psi
