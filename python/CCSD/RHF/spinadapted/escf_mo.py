import psi4
import numpy as np
psi4.core.be_quiet()
psi4.core.set_output_file("output.dat")

mol = psi4.geometry ( """
        O
        H 1 1.1
        H 1 1.1 2 104.0
        symmetry c1
        """)
enuc = mol.nuclear_repulsion_energy()

psi4.set_options({'basis':'sto-3g',
                  'scf_type':'pk'})

e,wfn = psi4.energy('scf',mol=mol,return_wfn=True)
mints = psi4.core.MintsHelper(wfn.basisset())

ea = wfn.epsilon_a().to_array()
nbf = len(ea)
_Ca = wfn.Ca()
Ca = _Ca.to_array()
mo_eri = mints.mo_eri(_Ca,_Ca,_Ca,_Ca)
_h = wfn.H().to_array()
h = np.zeros_like(_h)
ndocc = wfn.nalpha()

#for p in range(nbf):
#    for q in range(nbf):
#        tsum = 0.0
#        for m in range(nbf):
#            for n in range(nbf):
#                tsum += Ca[m][q]*Ca[n][p]*_h[m][n]
#        h[p][q] = tsum

h = np.einsum('mq,np,mn->pq',Ca,Ca,_h)

escf = 2*np.einsum('ii->',h[:ndocc,:ndocc])

# +---- this for loop ... 
# V
#for i in range(ndocc):
#    for j in range(ndocc):
#        escf += 2*mo_eri.np[i][i][j][j] - mo_eri.np[i][j][i][j]
#
# +----- and this einsum do the same thing
# V
escf += 2*np.einsum('iijj->',mo_eri.np[:ndocc,:ndocc,:ndocc,:ndocc]) - np.einsum('ijij->',mo_eri.np[:ndocc,:ndocc,:ndocc,:ndocc])

print(escf + enuc)
print(e)
