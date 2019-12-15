import psi4
import numpy as np
psi4.core.be_quiet()

mol = psi4.geometry("""
        O
        H 1 1.1
        H 1 1.1 2 104.0
        symmetry c1
        """)

psi4.set_options({"basis":"sto-3g",
                  "scf_type":"pk"})

e,wfn = psi4.energy('hf',mol=mol,return_wfn=True)
mints = psi4.core.MintsHelper(wfn.basisset())
_C = wfn.Ca()
C = _C.to_array()
moeri = mints.mo_eri(_C,_C,_C,_C)
moeri = moeri.np
nocc = wfn.nalpha()
nmo = wfn.nmo()
nvir = nmo - nocc
eps = wfn.epsilon_a().np

o = slice(0,nocc)
v = slice(nocc,nmo)

delta = np.zeros_like(moeri[o,o,v,v])
for i in range(nocc):
    for j in range(nocc):
        for a in range(nvir):
            for b in range(nvir):
                aa = a + nocc
                bb = b + nocc
                delta[i][j][a][b] = eps[i] + eps[j] - eps[aa] - eps[bb]

#transform to physicists notation : should not be required for (ab||ij) notation?
moeri = moeri.transpose(0,2,1,3)

emp2 = 0.0
eri_aa = moeri - moeri.transpose(0,1,3,2)
a_aa = -eri_aa[o,o,v,v]/delta
emp2 += (1/2)*np.einsum('ijab,ijab->',a_aa,eri_aa[o,o,v,v])


eri_ab = -moeri.transpose(0,1,3,2)
a_ab = -eri_ab[o,o,v,v]/delta
emp2 += np.einsum('ijab,ijab->',a_ab,eri_ab[o,o,v,v])
print(emp2)

