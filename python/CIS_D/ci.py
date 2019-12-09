import numpy as np
import psi4

def kron ( a , b ):
    assert type(a) == int, "Illegal value passed to kron."
    assert type(b) == int, "Illegal value passed to kron."
    return a == b
def HS ( F, eri, i, j, a, b ):
    "Hamiltoniam (H) matrix element for single excitations (S)"
    return kron(i,j)*F[a,b] - kron(a,b)*F[i,j] + eri[a,j,i,b]

if __name__ == "__main__":
    mol = psi4.geometry("""
    O 
    H 1 1.1
    H 1 1.1 2 104.0
    symmetry c1
    """)
    psi4.core.be_quiet()
    psi4.set_options({'basis':'sto-3g',
                           'reference':'rhf',
                           'scf_type':'pk'})
    e,wfn = psi4.energy('hf',mol=mol,return_wfn=True)
    ndocc = wfn.nalpha()
    h = wfn.H().to_array()
    C = wfn.Ca()
    mints = psi4.core.MintsHelper(wfn.basisset())
    eri = mints.mo_eri(C,C,C,C).to_array()


    #form fock matrix in MO basis
    f =  np.zeros_like(h)
    f += h
    f += 2*np.einsum('pqkk->pq',eri[:,:,:ndocc,:ndocc])
    f -= np.einsum('pkqk->pq',eri[:,:ndocc,:,:ndocc])

    print(HS(f,eri,0,1,0,3))
