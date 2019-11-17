#Configuration Interaction Singles
import psi4
import numpy as np
import time

mol = psi4.geometry("""
    O
    H 1 0.96
    H 1 0.96 2 104.5
    symmetry c1""")

psi4.core.be_quiet()
psi4.set_options({'basis':'cc-pvtz',
                  'scf_type':'pk'})
scf_e,wfn = psi4.energy('scf',return_wfn=True)
mints = psi4.core.MintsHelper(wfn.basisset())
C = wfn.Ca()
ndocc = wfn.doccpi()[0]
nmo = wfn.nmo()
nvirt = nmo - ndocc
nDet_S = ndocc * nvirt * 2

#generate one electron integrals
H = np.asarray(mints.ao_kinetic()) + np.asarray(mints.ao_potential())

#Make spin orbital MO
MO = np.asarray(mints.mo_spin_eri(C,C))
H = np.einsum('uj,vi,uv', C, C, H)
H = np.repeat(H, 2, axis=0)
H = np.repeat(H, 2, axis=1)

spin_ind = np.arange(H.shape[0], dtype=np.int) % 2
H *= (spin_ind.reshape(-1,1) == spin_ind)

from helper_CI import Determinant, HamiltonianGenerator
from itertools import combinations

#generate singly sub determinants
occList = [i for i in range(ndocc)]
det_ref = Determinant(alphaObtList=occList, betaObtList=occList)
detList = det_ref.generateSingleExcitationsOfDet(nmo)
detList.append(det_ref)

Hamiltonian_generator = HamiltonianGenerator(H, MO)
Hamiltonian_matrix = Hamiltonian_generator.generateMatrix(detList)

t1 = time.time()
ecis,wfns = np.linalg.eigh(Hamiltonian_matrix)
dt = time.time() - t1
hartree2eV = 27.211
print('\nCIS Excitation Energies (Singlets only):')
print(' #        Hartree                  eV')
print('--  --------------------  --------------------')
for i in range(1, len(ecis)):
    excit_e = ecis[i] + mol.nuclear_repulsion_energy() - scf_e
    print('%2d %20.10f %20.10f' % (i, excit_e, excit_e * hartree2eV))

print('Diagonalization time: {}'.format(dt))
