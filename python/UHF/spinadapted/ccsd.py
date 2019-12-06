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
        0 1
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
mo_ijab = mints.mo_eri(_Ca,_Ca,_Ca,_Ca).to_array()
mo_iJaB = mints.mo_eri(_Ca,_Cb,_Ca,_Cb).np
mo_iJAb = mints.mo_eri(_Ca,_Cb,_Cb,_Ca).np
mo_IJAB = mints.mo_eri(_Cb,_Cb,_Cb,_Cb).np
mo_IjAb = mints.mo_eri(_Cb,_Ca,_Cb,_Ca).np
mo_IjaB = mints.mo_eri(_Cb,_Ca,_Ca,_Cb).np
mo_ijAB = mints.mo_eri(_Ca,_Ca,_Cb,_Cb).np
mo_IJab = mints.mo_eri(_Cb,_Cb,_Ca,_Ca).np

escf = 0.0
escf = np.einsum('ii->',ha[oa,oa]) + np.einsum('ii->',hb[ob,ob])
print(0.5*np.einsum('iijj',mo_ijab[oa,oa,oa,oa]))
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

#form antisymmetrized MO integrals
ijab =  mo_ijab.transpose(0,2,1,3) - mo_ijab.transpose(0,3,2,1)
iJaB =  mo_iJaB.transpose(0,2,1,3)
iJAb = -mo_iJAb.transpose(0,3,1,2)
IJAB =  mo_IJAB.transpose(0,2,1,3) - mo_IJAB.transpose(0,3,2,1)
IjAb =  mo_IjAb.transpose(0,2,1,3)
IjaB = -mo_IjaB.transpose(0,3,1,2)


Dia = np.zeros((nalpha,nbf-nalpha))
DIA = np.zeros((nbeta,nbf-nbeta))

for i in range(nalpha):
    for a in range(nbf-nalpha):
        aa = a + nalpha
        Dia[i][a] = fa[i][i] - fa[aa][aa]
for i in range(nbeta):
    for a in range(nbf-nbeta):
        aa = a + nbeta
        DIA[i][a] = fb[i][i] - fb[aa][aa]

tia = fa[oa,va]/Dia
tIA = fb[ob,vb]/DIA

occa = nalpha
vira = nbf - nalpha
occb = nbeta
virb = nbf - nbeta

Dijab = np.zeros((occa,occa,vira,vira))
DiJaB = np.zeros((occa,occb,vira,virb))
DiJAb = np.zeros((occa,occb,virb,vira))
DIJAB = np.zeros((occb,occb,virb,virb))
DIjAb = np.zeros((occb,occa,virb,vira))
DIjaB = np.zeros((occb,occa,vira,virb))

for i in range(occa):
    for j in range(occa):
        for a in range(vira):
            for b in range(vira):
                aa = a + nalpha
                bb = b + nalpha
                Dijab[i][j][a][b] = fa[i][i] + fa[j][j] - fa[aa][aa] - fa[bb][bb]
tijab = ijab[oa,oa,va,va]/Dijab

for i in range(occa):
    for j in range(occb):
        for a in range(vira):
            for b in range(virb):
                aa = a + nalpha
                bb = b + nbeta
                DiJaB[i][j][a][b] = fa[i][i] + fb[j][j] - fa[aa][aa] - fb[bb][bb]
tiJaB = iJaB[oa,ob,va,vb]/DiJaB

for i in range(occa):
    for j in range(occb):
        for a in range(virb):
            for b in range(vira):
                aa = a + nbeta
                bb = b + nalpha
                DiJAb[i][j][a][b] = fa[i][i] + fb[j][j] - fb[aa][aa] - fa[bb][bb]
tiJAb = iJAb[oa,ob,vb,va]/DiJAb

for i in range(occb):
    for j in range(occb):
        for a in range(virb):
            for b in range(virb):
                aa = a + nbeta
                bb = b + nbeta
                DIJAB[i][j][a][b] = fb[i][i] + fb[j][j] - fb[aa][aa] - fb[bb][bb]
tIJAB = IJAB[ob,ob,vb,vb]/DIJAB

for i in range(occb):
    for j in range(occa):
        for a in range(virb):
            for b in range(vira):
                aa = a + nbeta
                bb = b + nalpha
                DIjAb[i][j][a][b] = fb[i][i] + fa[j][j] - fb[aa][aa] - fa[bb][bb]
tIjAb = IjAb[ob,oa,vb,va]/DIjAb

for i in range(occb):
    for j in range(occa):
        for a in range(vira):
            for b in range(virb):
                aa = a + nalpha
                bb = b + nbeta
                DIjaB[i][j][a][b] = fb[i][i] + fa[j][j] - fa[aa][aa] - fb[bb][bb]
tIjaB = IjaB[ob,oa,va,vb]/DIjaB

emp2 = 0
emp2 += (1/4)*np.einsum('ijab,ijab->',tijab,ijab[oa,oa,va,va])
emp2 +=       np.einsum('ijab,ijab->',tiJaB,iJaB[oa,ob,va,vb])
emp2 += (1/4)*np.einsum('ijab,ijab->',tIJAB,IJAB[ob,ob,vb,vb])
print(emp2)