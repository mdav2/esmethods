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
nalpha = wfn.nalpha()
nbeta = wfn.nbeta()
nbf = wfn.nmo()
occa = nalpha
vira = nbf - nalpha
occb = nbeta
virb = nbf - nbeta
oa = slice(0,nalpha)
va = slice(nalpha,nbf)
ob = slice(0,nbeta)
vb = slice(nbeta,nbf)

_Ca = wfn.Ca()
_Cb = wfn.Cb()
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

#compute scf energy
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

#form antisymmetrized MO integrals
# 0 1 2 3
# i j a b
# (0,2,1,3) (0,1,3,2)
ijab =  mo_ijab.transpose(0,2,1,3) - mo_ijab.transpose(0,2,1,3).transpose(0,1,3,2)
iJaB =  mo_iJaB.transpose(0,2,1,3)
iJAb = -mo_iJAb.transpose(0,3,1,2)
IJAB =  mo_IJAB.transpose(0,2,1,3) - mo_IJAB.transpose(0,2,1,3).transpose(0,1,3,2)
IjAb =  mo_IjAb.transpose(0,2,1,3)
IjaB = -mo_IjaB.transpose(0,3,1,2)


#form 2index D arrays
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

#form 4index D arrays
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
print("MP2 correction ", emp2)

def kron(a,b):
    assert type(a) == int
    assert type(b) == int
    return a == b

def form_Fae(fa,tia,tIA,tijab,tiJaB,tIjaB,ijab,iJaB,IjaB):
    #expansion of (tia)(tia)
    tiatia = np.einsum('ma,nf->mnaf',tia,tia)
    #expansion of (tia)(tIA)
    tiatIA = np.einsum('ma,nf->mnaf',tia,tIA)
    #expansion of (tIA)(tia)
    tIAtia = np.einsum('ma,nf->mnaf',tIA,tia)
    #term 1 
    Fae = np.zeros_like(fa[va,va])
    Fae += fa[va,va]
    Fae -= np.diagonal(fa[va,va])
    #term 2 
    Fae -= (1/2)*np.einsum('me,ma->ae',fa[oa,va],tia)
    #term 3
    Fae += np.einsum('mf,amef->ae',tia,ijab[va,oa,va,va])
    #term 4
    Fae += np.einsum('mf,amef->ae',tIA,IJAB[vb,ob,vb,vb])
    #term 5
    Fae -= (1/2)*np.einsum('mnaf,mnef->ae',(tijab + (1/2)*(tiatia - tiatia.transpose(0,1,3,2))),ijab[oa,oa,va,va])
    #term 6
    Fae -= (1/2)*np.einsum('mnaf,mnef->ae',(tiJaB + (1/2)*tiatIA), iJaB[oa,ob,va,vb])
    #term 7
    Fae -= (1/2)*np.einsum('mnaf,mnef->ae',(tIjaB - (1/2)*tIAtia),IjaB[ob,oa,va,vb])
    return Fae 

def form_FAE(fb,tia,tIA,tIJAB,tIjAb,tiJaB,IJAB,IjAb,iJAb):
    #expansion of (tIA)(tIA)
    tIAtIA = np.einsum('ma,nf->mnaf',tIA,tIA)
    #expansion of (tIA)(tia)
    tIAtia = np.einsum('ma,nf->mnaf',tIA,tia)
    #expansion of (tia)(tIA)
    tiatIA = np.einsum('ma,nf->mnaf',tia,tIA)
    #term 1
    FAE = np.zeros_like(fb[vb,vb])
    FAE += fb[va,va]
    FAE -= np.diagonal(fb[vb,vb])
    #term 2
    FAE -= (1/2)*np.einsum('me,ma->ae',fb[ob,vb],tIA)
    #term 3
    FAE += np.einsum('mf,amef->ae',tIA,IJAB[vb,ob,vb,vb])
    #term 4
    FAE += np.einsum('mf,amef->ae',tia,ijab[va,oa,va,va])
    #term 5
    FAE -= (1/2)*np.einsum('mnaf,mnef->ae',(tiJaB + (1/2)*(tIAtIA - tIAtIA.transpose(0,1,3,2))),IJAB[ob,ob,vb,vb])
    #term 6
    FAE -= (1/2)*np.einsum('mnaf,mnef->ae',(tIjAb + (1/2)*tIAtia),IjAb[ob,oa,vb,va])
    #term 7
    FAE -= (1/2)*np.einsum('mnaf,mnef->ae',(tiJAb - (1/2)*(tiatIA)),iJAb[oa,ob,vb,va])
    return Fae

def form_Fmi(fa,tia,tIA,tijab,tiJaB,tiJAb,ijab,iJaB,iJAb):
    #expansion of (tia)(tia)
    tiatia = np.einsum('ie,nf->inef',tia,tia)
    #expansion of (tia)(tIA)
    tiatIA = np.einsum('ie,nf->inef',tia,tIA)
    #term 1
    Fmi = np.zeros_like(fa[oa,oa])
    Fmi += fa[oa,oa]
    Fmi -= np.diagonal(fa[oa,oa])
    #term 2
    Fmi += (1/2)*np.einsum('me,ie->mi',fa[oa,va],tia)
    #term 3
    Fmi += np.einsum('ne,mnie->mi',tia,ijab[oa,oa,oa,va])
    #term 4
    Fmi += np.einsum('ne,mnie->mi',tIA,iJaB[oa,ob,oa,vb])
    #term 5
    Fmi += (1/2)*np.einsum('inef,mnef->mi',(tijab + (1/2)*(tiatia - tiatia.transpose(0,1,3,2))),ijab[oa,oa,va,va])
    #term 6
    Fmi += (1/2)*np.einsum('inef,mnef->mi',(tiJaB + (1/2)*tiatIA), iJaB[oa,ob,va,vb])
    #term 7
    Fmi += (1/2)*np.einsum('inef,mnef->mi',(tiJAb - (1/2)*tiatIA.transpose(0,1,3,2) ), iJAb[oa,ob,vb,va])
    return Fmi

def form_FMI(fb,tia,tIA,tIJAB,tIjAb,tIjaB,IJAB,IjAb,IjaB):
    #expansion of (tIA)(tia)
    tIAtia = np.einsum('ie,nf->inef',tIA,tia)
    #expansion of (tIA)(tIA)
    tIAtIA = np.einsum('ie,nf->inef',tIA,tIA)
    #term 1
    FMI = np.zeros_like(fb[ob,ob])
    FMI += fb[ob,ob]
    FMI -= np.diagonal(fb[ob,ob])
    #term 2
    FMI += (1/2)*np.einsum('me,ie->mi',fb[ob,vb],tIA)
    #term 3
    FMI += np.einsum('ne,mnie->mi',tIA,IJAB[ob,ob,ob,vb])
    #term 4
    FMI += np.einsum('ne,mnie->mi',tia,IjAb[ob,oa,ob,va])
    #term 5
    FMI += (1/2)*np.einsum('inef,mnef->mi',tIJAB + (1/2)*(tIAtIA - tIAtIA.transpose(0,1,3,2)),IJAB[ob,ob,vb,vb])
    #term 6
    FMI += (1/2)*np.einsum('inef,mnef->mi',tIjAb + (1/2)*tIAtia, IjAb[ob,oa,vb,va])
    #term 7
    FMI += (1/2)*np.einsum('inef,mnef->mi',tIjaB - (1/2)*tIAtia, IjaB[ob,oa,va,vb])
    return FMI

def form_Fme(fa,tia,tIA,ijab,iJaB):
    #term 1
    Fme = np.zeros_like(fa[oa,va])
    Fme += fa[oa,va]
    #term 2
    Fme += np.einsum('nf,mnef->me',tia,ijab[oa,oa,va,va])
    #term 3
    Fme += np.einsum('nf,mnef->me',tIA,iJaB[oa,ob,va,vb])
    return Fme

def form_FME(fb,tia,tIA,IJAB,IjAb):
    #term 1
    FME = np.zeros_like(fb[ob,vb])
    #term 2
    FME += fb[ob,vb]
    #term 3
    FME += np.einsum('nf,mnef->me',tIA,IJAB[ob,ob,vb,vb])
    #term 4
    FME += np.einsum('nf,mnef->me',tia,IjAb[ob,oa,vb,va])

    return FME

def form_Wmnij(tia,tijab,ijab):
    #expansion of (tia)(tia)
    tiatia = np.einsum('ie,jf->ijef',tia,tia)
    Wmnij = np.zeros_like(ijab[oa,oa,oa,oa])
    #term 1
    Wmnij += np.einsum('je,mnie->mnij',tia,ijab[oa,oa,oa,va])
    #term 2
    Wmnij -= np.einsum('ie,mnje->mnij',tia,ijab[oa,oa,oa,va])
    #term 3
    Wmnij += (1/4)*np.einsum('ijef,mnef->mnij',tijab + tiatia - tiatia.transpose(0,1,3,2),ijab[oa,oa,va,va])
    return Wmnij

def form_WmNiJ(tia,tIA,tiJaB,tiJAb,iJaB,iJAb):
    #expansion of (tia)(tIA)
    tiatIA = np.einsum('ie,jf->ijef',tia,tIA)

    WmNiJ = np.zeros_like(iJaB[oa,ob,oa,ob])
    #term 1
    WmNiJ += iJaB[oa,ob,oa,ob]
    #term 2
    WmNiJ += np.einsum('je,mnie->mnij',tIA,iJaB[oa,ob,oa,vb])
    #term 3
    WmNiJ -= np.einsum('ie,mnje->mnij',tia,iJAb[oa,ob,ob,va])
    #term 4
    WmNiJ += (1/4)*np.einsum('ijef,mnef->mnij',tiJaB + tiatIA, iJaB[oa,ob,va,vb])
    #term 5
    WmNiJ += (1/4)*np.einsum('ijef,mnef->mnij',tiJAb + tiatIA.transpose(0,1,3,2),iJAb[oa,ob,vb,va])
    return WmNiJ

def form_WmNIj(tia,tIA,tIjAb,tIjaB,iJAb,iJaB):
    #expansion of (tIA)(tia)
    tIAtia = np.einsum('ie,jf->ijef',tIA,tia)

    WmNIj = np.zeros_like(iJAb[oa,ob,ob,oa])
    #term 1
    WmNIj += iJaB[oa,ob,ob,oa]
    #term 2
    WmNIj += np.einsum('je,mnie->mnij',tia,iJAb[oa,ob,ob,va])
    #term 3
    WmNIj -= np.einsum('ie,mnje->mnij',tIA,iJaB[oa,ob,oa,vb])
    #term 4
    WmNIj += (1/4)*np.einsum('ijef,mnef->mnij',tIjAb + tIAtia,iJAb[oa,ob,vb,va])
    #term 5
    WmNIj += (1/4)*np.einsum('ijef,mnef->mnij',tIjaB - tIAtia.transpose(0,1,3,2),iJaB[oa,ob,va,vb])
    return WmNIj

def form_WMNIJ(tIA,tIJAB,IJAB):
    #expansion of (tIA)(tIA)
    tIAtIA = np.einsum('ie,jf->ijef',tIA,tIA)

    WMNIJ = np.zeros_like(IJAB[ob,ob,ob,ob])
    #term 1
    WMNIJ += IJAB[ob,ob,ob,ob]
    #term 2
    WMNIJ += np.einsum('je,mnie->mnij',tIA,IJAB[ob,ob,ob,vb])
    #term 3
    WMNIJ -= np.einsum('ie,mnje->mnij',tIA,IJAB[ob,ob,ob,vb])
    #term 4
    WMNIJ += (1/4)*np.einsum('ijef,mnef->mnij',tIJAB + tIAtIA + tIAtIA.transpose(0,1,3,2),IJAB[ob,ob,vb,vb])
    return WMNIJ

def form_WMnIj(tia,tIA,tIjAb,tIjaB,IjAb,IjaB):
    #expansion of (tIA)(tia)
    tIAtia = np.einsum('ie,jf->ijef',tIA,tia)

    WMnIj = np.zeros_like(IjAb[ob,oa,ob,oa])
    #term 1
    WMnIj += IjAb[ob,oa,ob,oa]
    #term 2
    WMnIj += np.einsum('je,mnie->mnij',tia,IjAb[ob,oa,ob,va])
    #term 3
    WMnIj -= np.einsum('ie,mnje->mnij',tIA,IjaB[ob,oa,oa,vb])
    #term 4
    WMnIj += (1/4)*np.einsum('ijef,mnef->mnij',tIjAb + tIAtia,IjAb[ob,oa,vb,va])
    #term 5
    WMnIj += (1/4)*np.einsum('ijef,mnef->mnij',tIjaB - tIAtia,IjaB[ob,oa,va,vb])
    return WMnIj

def form_WMniJ(tia,tIA,tIjAb,tiJAb,IjaB,IjAb):
    #expansion of (tia)(tIA)
    tiatIA = np.einsum('ie,jf->ijef',tia,tIA)

    WMniJ = np.zeros_like(IjaB[ob,oa,oa,ob])
    #term 1
    WMniJ += IjaB[ob,oa,oa,ob]
    #term 2
    WMniJ += np.einsum('je,mnie->mnij',tIA,IjaB[ob,oa,oa,vb])
    #term 3
    WMniJ -= np.einsum('ie,mnje->mnij',tia,IjAb[ob,oa,ob,va])
    #term 4
    WMniJ += (1/4)*np.einsum('ijef,mnef->mnij',tiJaB + tiatIA,IjaB[ob,oa,va,vb])
    #term 5
    WMniJ += (1/4)*np.einsum('ijef,mnef->mnij',tiJAb - tiatIA, IjAb[ob,oa,vb,va])

    return WMniJ

def form_Wabef(tia,tijab,ijab):
    #expansion of (tia)(tia)
    tiatia = np.einsum('ma,nb->mnab',tia,tia)

    Wabef = np.zeros_like(ijab[va,va,va,va])
    #term 1
    Wabef += ijab[va,va,va,va]
    #term 2
    Wabef -= np.einsum('mb,amef->abef',tia,ijab[va,oa,va,va])
    #term 3
    Wabef += np.einsum('ma,bmef->abef',tia,ijab[va,oa,va,va])
    #term 4
    Wabef += (1/4)*np.einsum('mnab,mnef->abef',tijab + tiatia - tiatia.transpose(0,1,3,2),ijab[oa,oa,va,va])
    return Wabef

Fae = form_Fae(fa,tia,tIA,tijab,tiJaB,tIjaB,ijab,iJaB,IjaB)
FAE = form_FAE(fb,tia,tIA,tIJAB,tIjAb,tiJaB,IJAB,IjAb,iJAb)
Fmi = form_Fmi(fa,tia,tIA,tijab,tiJaB,tiJAb,ijab,iJaB,iJAb)
FMI = form_FMI(fb,tia,tIA,tIJAB,tIjAb,tIjaB,IJAB,IjAb,IjaB)
Fme = form_Fme(fa,tia,tIA,ijab,iJaB)
FME = form_FME(fb,tia,tIA,IJAB,IjAb)

Wmnij = form_Wmnij(tia,tijab,ijab)
WmNiJ = form_WmNiJ(tia,tIA,tiJaB,tiJAb,iJaB,iJAb)
WmNIj = form_WmNIj(tia,tIA,tIjAb,tIjaB,iJAb,iJaB)
WMNIJ = form_WMNIJ(tIA,tIJAB,IJAB)
WMnIj = form_WMnIj(tia,tIA,tIjAb,tIjaB,IjAb,IjaB)
WMniJ = form_WMniJ(tia,tIA,tIjAb,tiJAb,IjaB,IjAb)

Wabef = form_Wabef(tia,tijab,ijab)
