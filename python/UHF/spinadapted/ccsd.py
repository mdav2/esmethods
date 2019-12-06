import psi4
import numpy as np
from time import time
from copy import deepcopy
np.set_printoptions(precision=6, linewidth=200, suppress=True)
psi4.core.be_quiet()
psi4.core.set_num_threads(4)
psi4.core.set_output_file("output.dat")
t1p = 0
t2p = 0
tecc = 0 
tmem = 0

mol = psi4.geometry ( """
        1 2
        O
        H 1 1.1
        H 1 1.1 2 104.0
        symmetry c1
        """)
enuc = mol.nuclear_repulsion_energy()

psi4.set_options({'basis':'sto-3g',
                  'reference':'uhf',
                  'e_convergence':1e-12,
                  'd_convergence':1e-10,
                  'scf_type':'pk'})
e,wfn = psi4.energy('scf',return_wfn=True,mol=mol)
_Ca = wfn.Ca()
_Cb = wfn.Cb()
nalpha = wfn.nalpha()
nbeta = wfn.nbeta()
nbf = wfn.nmo()
oa = slice(0,nalpha)
va = slice(nalpha,nbf)
ob = slice(0,nbeta)
vb = slice(nbeta,nbf)
Ca = _Ca.to_array()
Cb = _Cb.to_array()
H = wfn.H().to_array()
ha = np.einsum('mq,np,mn->pq',Ca,Ca,H)
hb = np.einsum('mq,np,mn->pq',Cb,Cb,H)
mints = psi4.core.MintsHelper(wfn.basisset())
mo_ijab = mints.mo_eri(_Ca,_Ca,_Ca,_Ca).np
mo_iJaB = mints.mo_eri(_Ca,_Cb,_Ca,_Cb).np
mo_iJAb = mints.mo_eri(_Ca,_Cb,_Cb,_Ca).np
mo_IJAB = mints.mo_eri(_Cb,_Cb,_Cb,_Cb).np
mo_IjAb = mints.mo_eri(_Cb,_Ca,_Cb,_Ca).np
mo_IjaB = mints.mo_eri(_Cb,_Ca,_Ca,_Cb).np
mo_ijAB = mints.mo_eri(_Ca,_Ca,_Cb,_Cb).np
mo_IJab = mints.mo_eri(_Cb,_Cb,_Ca,_Ca).np
escf = 0.0
escf = np.einsum('ii->',ha[oa,oa]) + np.einsum('ii->',hb[ob,ob])
escf += 0.5*np.einsum('iijj',mo_ijab[oa,oa,oa,oa]) #coulomb, alpha-alpha
escf += 0.5*np.einsum('iijj',mo_IJAB[ob,ob,ob,ob]) #coulomb, beta-beta
escf += np.einsum('iijj',mo_ijAB[oa,oa,ob,ob])     #coulomb, alpha-beta
escf -= 0.5*np.einsum('ijji',mo_IJAB[ob,ob,ob,ob]) #exchange, beta-beta
escf -= 0.5*np.einsum('ijji',mo_ijab[oa,oa,oa,oa]) #exchange, alpha-alpha
print(escf+enuc)

#make fock matrices
fa = ha + np.einsum('pqkk->pq',mo_ijab[:,:,oa,oa]) - np.einsum('pkkq->pq',mo_ijab[:,oa,oa,:])\
        + np.einsum('pqkk->pq',mo_ijAB[:,:,ob,ob])
fb = hb + np.einsum('pqkk->pq',mo_IJAB[:,:,ob,ob]) - np.einsum('pkkq->pq',mo_IJAB[:,ob,ob,:])\
        + np.einsum('pqkk->pq',mo_IJab[:,:,oa,oa])