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
    return FAE

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

def form_WaBeF(tia,tIA,tiJaB,tIjaB,iJaB,IjaB):
    #expansion of (tia)(tIA)
    tiatIA = np.einsum('ma,nb->mnab',tia,tIA)
    #expansion of (tIA)(tia)
    tIAtia = np.einsum('ma,nb->mnab',tIA,tia)
    WaBeF = np.zeros_like(iJaB[va,vb,va,vb])
    #term 1
    WaBeF += iJaB[va,vb,va,vb]
    #term 2
    WaBeF -= np.einsum('mb,amef->abef',tIA,iJaB[va,ob,va,vb])
    #term 3
    WaBeF += np.einsum('ma,bmef->abef',tia,IjaB[vb,oa,va,vb])
    #term 4
    WaBeF += (1/4)*np.einsum('mnab,mnef->abef',tiJaB + tiatIA,iJaB[oa,ob,va,vb])
    #term 5
    WaBeF += (1/4)*np.einsum('mnab,mnef->abef',tIjaB - tIAtia.transpose(0,1,3,2),IjaB[ob,oa,va,vb])
    return WaBeF

def form_WaBEf(tia,tIA,tiJaB,tIjaB,iJaB,IjAb):
    #expansion of (tia)(tIA)
    tiatIA = np.einsum('ma,nb->mnab',tia,tIA)
    #expansion of (tIA)(tia)
    tIAtia = np.einsum('ma,nb->mnab',tIA,tia)

    WaBEf = np.zeros_like(iJAb[va,vb,vb,va])
    #term 1
    WaBEf += iJAb[va,vb,vb,va]
    #term 2
    WaBEf -= np.einsum('mb,amef->abef',tIA,iJAb[va,ob,vb,va])
    #term 3
    WaBEf += np.einsum('ma,bmef->abef',tia,IjAb[vb,oa,vb,va])
    #term 4
    WaBEf += (1/4)*np.einsum('mnab,mnef->abef',tiJaB + tiatIA,iJAb[oa,ob,vb,va])
    #term 5
    WaBEf += (1/4)*np.einsum('mnab,mnef->abef',tIjaB - tIAtia.transpose(0,1,3,2),IjAb[ob,oa,vb,va])
    return WaBEf

def form_WABEF(tIA,tIJAB,IJAB):
    #expansion of tIAtIA
    tIAtIA = np.einsum('ma,nb->mnab',tIA,tIA)

    WABEF = np.zeros_like(IJAB[vb,vb,vb,vb])
    #term 1
    WABEF += IJAB[vb,vb,vb,vb]
    #term 2
    WABEF -= np.einsum('mb,amef->abef',tIA,IJAB[vb,ob,vb,vb])
    #term 3
    WABEF += np.einsum('ma,bmef->abef',tIA,IJAB[vb,ob,vb,vb])
    #term 4
    WABEF += (1/4)*np.einsum('mnab,mnef->abef',tIJAB + tIAtIA - tIAtIA.transpose(0,1,3,2), IJAB[ob,ob,vb,vb])
    return WABEF

def form_WAbEf(tia,tIA,tIjAb,tiJAb,IjAb,iJAb):
    #expansion of (tIA)(tia)
    tIAtia = np.einsum('ma,nb->mnab',tIA,tia)
    #expansion of (tia)(tIA)
    tiatIA = np.einsum('ma,nb->mnab',tia,tIA)

    WAbEf = np.zeros_like(IjAb[vb,va,vb,va])

    #term 1
    WAbEf += IjAb[vb,va,vb,va]
    #term 2
    WAbEf -= np.einsum('mb,amef->abef',tia,IjAb[vb,oa,vb,va])
    #term 3
    WAbEf += np.einsum('ma,bmef->abef',tIA,iJAb[va,ob,vb,va])
    #term 4
    WAbEf += (1/4)*np.einsum('mnab,mnef->abef',tIjAb + tIAtia,IjAb[ob,oa,vb,va])
    #term 5
    WAbEf += (1/4)*np.einsum('mnab,mnef->abef',tiJAb - tiatIA.transpose(0,1,3,2),iJAb[oa,ob,vb,va])
    return WAbEf

def form_WAbeF(tia,tIA,tIjAb,tijab,IjaB,iJaB):
    #expansion of (tIA)(tia)
    tIAtia = np.einsum('ma,nb->mnab',tIA,tia)
    #expansion of (tia)(tIA)
    tiatIA = np.einsum('ma,nb->mnab',tia,tIA)

    WAbeF = np.zeros_like(IjaB[vb,va,va,vb])
    #term 1
    WAbeF += IjaB[vb,va,va,vb]
    #term 2
    WAbeF -= np.einsum('mb,amef->abef',tia,IjaB[vb,oa,va,vb])
    #term 3
    WAbeF += np.einsum('ma,bmef->abef',tIA,iJaB[va,ob,va,vb])
    #term 4
    WAbeF += (1/4)*np.einsum('mnab,mnef->abef',tIjAb + tIAtia,IjaB[ob,oa,va,vb])
    #term 5
    WAbeF += (1/4)*np.einsum('mnab,mnef->abef',tiJAb - tiatIA.transpose(0,1,3,2),iJaB[oa,ob,va,vb])
    return WAbeF

def form_Wmbej(tia,tijab,tiJAb,ijab,iJaB):
    #expansion of (tia)(tia)
    tiatia = np.einsum('jf,nb->jnfb',tia,tia)

    Wmbej = np.zeros_like(ijab[oa,va,va,oa])
    #term 1
    Wmbej += ijab[oa,va,va,oa]
    #term 2
    Wmbej += np.einsum('jf,mbef->mbej',tia,ijab[oa,va,va,va])
    #term 3
    Wmbej -= np.einsum('nb,mnej->mbej',tia,ijab[oa,oa,va,oa])
    #term 4 
    Wmbej -= np.einsum('jnfb,mnef->mbej',(1/2)*tijab + tiatia,ijab[oa,oa,va,va])
    #term 5
    Wmbej -= (1/2)*np.einsum('jnfb,mnef->mbej',tiJAb,iJaB[oa,ob,va,vb])

    return Wmbej

def form_WmBeJ(tIA,tIJAB,tIjaB,iJaB,ijab):
    #expansion of (tIA)(tIA)
    tIAtIA = np.einsum('jf,nb->jnfb',tIA,tIA)

    WmBeJ = np.zeros_like(iJaB[oa,vb,va,ob])
    #term 1
    WmBeJ += iJaB[oa,vb,va,ob]
    #term 2
    WmBeJ += np.einsum('jf,mbef->mbej',tIA,iJaB[oa,vb,va,vb])
    #term 3
    WmBeJ -= np.einsum('nb,mnej->mbej',tIA,iJaB[oa,ob,va,ob])
    #term 4
    WmBeJ -= np.einsum('jnfb,mnef->mbej',(1/2)*tIJAB + tIAtIA, iJaB[oa,ob,va,vb])
    #term 5
    WmBeJ -= (1/2)*np.einsum('jnfb,mnef->mbej',tIjaB,ijab[oa,oa,va,va])
    return WmBeJ

def form_WmBEj(tia,tIA,tiJaB,iJAb):
    #expansion of (tia)(tIA)
    tiatIA = np.einsum('jf,nb->jnfb',tia,tIA)

    WmBEj = np.zeros_like(iJAb[oa,vb,vb,oa])

    #term 1
    WmBEj += iJAb[oa,vb,vb,oa]
    #term 2
    WmBEj += np.einsum('jf,mbef->mbej',tia,iJAb[oa,vb,vb,va])
    #term 3
    WmBEj -= np.einsum('nb,mnej->mbej',tIA,iJAb[oa,ob,vb,oa])
    #term 4
    WmBEj -= np.einsum('jnfb,mnef->mbej',(1/2)*tiJaB + tiatIA, iJAb[oa,ob,vb,va])
    return WmBEj

def form_WMBEJ(tIA,tIJAB,IJAB,IjAb):
    #expansion of (tIA)(tIA)
    tIAtIA = np.einsum('jf,nb->jnfb',tIA,tIA)

    WMBEJ = np.zeros_like(IJAB[ob,vb,vb,ob])
    #term 1
    WMBEJ += IJAB[ob,vb,vb,ob]
    #term 2
    WMBEJ += np.einsum('jf,mbef->mbej',tIA,IJAB[ob,vb,vb,vb])
    #term 3
    WMBEJ -= np.einsum('nb,mnej->mbej',tIA,IJAB[ob,ob,vb,ob])
    #term 4
    WMBEJ -= np.einsum('jnfb,mnef->mbej',(1/2)*tIJAB + tIAtIA, IJAB[ob,ob,vb,vb])
    #term 5
    WMBEJ -= (1/2)*np.einsum('jnfb,mnef->mbej', tIjaB, IjAb[ob,oa,vb,va])
    return WMBEJ

def form_WMbEj(tia,tijab,tiJAb,IjAb,IJAB):
    #expansion of (tia)(tia)
    tiatia = np.einsum('jf,nb->jnfb',tia,tia)

    WMbEj = np.zeros_like(IjAb[ob,va,vb,oa])
    #term 1
    WMbEj += IjAb[ob,va,vb,oa]
    #term 2
    WMbEj += np.einsum('jf,mbef->mbej',tia,IjAb[ob,va,vb,va])
    #term 3
    WMbEj -= np.einsum('nb,mnej->mbej',tia,IjAb[ob,oa,vb,oa])
    #term 4
    WMbEj -= np.einsum('jnfb,mnef->mbej',(1/2)*tijab + tiatia, IjAb[ob,oa,vb,va])
    #term 5
    WMbEj -= (1/2)*np.einsum('jnfb,mnef->mbej',tiJAb,IJAB[ob,ob,vb,vb])
    return WMbEj

def form_WMbeJ(tia,tIA,tIjAb,IjaB):
    #expansion of (tIA)(tia)
    tIAtia = np.einsum('jf,nb->jnfb',tIA,tia)

    WMbeJ = np.zeros_like(IjaB[ob,va,va,ob])
    #term 1
    WMbeJ += IjaB[ob,va,va,ob]
    #term 2
    WMbeJ += np.einsum('jf,mbef->mbej',tIA,IjaB[ob,va,va,vb])
    #term 3
    WMbeJ -= np.einsum('nb,mnej->mbej',tia,IjaB[ob,oa,va,ob])
    #term 4
    WMbeJ -= np.einsum('jnfb,mnef->mbej',(1/2)*tIjAb + tIAtia,IjaB[ob,oa,va,vb])
    return WMbeJ

def update_tia(fa,tia,tIA,tijab,tiJaB,tIjaB,tiJAb,ijab,iJaB,IjaB,iJAb,Dia):
    # one particle intermediates
    Fae = form_Fae(fa,tia,tIA,tijab,tiJaB,tIjaB,ijab,iJaB,IjaB)
    Fmi = form_Fmi(fa,tia,tIA,tijab,tiJaB,tiJAb,ijab,iJaB,iJAb)
    Fme = form_Fme(fa,tia,tIA,ijab,iJaB)
    FME = form_FME(fb,tia,tIA,IJAB,IjAb)

    _tia = np.zeros_like(tia)
    #term 1
    _tia += fa[oa,va]
    #term 2
    _tia += np.einsum('ie,ae->ia',tia,Fae)
    #term 3
    _tia -= np.einsum('ma,mi->ia',tia,Fmi)
    #term 4
    _tia += np.einsum('imae,me->ia',tijab,Fme)
    #term 5
    _tia += np.einsum('imae,me->ia',tiJaB,FME)
    #term 6
    _tia += np.einsum('me,amie->ia',tia,ijab[va,oa,oa,va])
    #term 7
    _tia += np.einsum('me,amie->ia',tIA,iJaB[va,ob,oa,vb])
    #term 8
    _tia -= (1/2)*np.einsum('mnae,mnie->ia',tijab,ijab[oa,oa,oa,va])
    #term 9
    _tia -= (1/2)*np.einsum('mnae,mnie->ia',tiJaB,iJaB[oa,ob,oa,vb])
    #term 10
    _tia -= (1/2)*np.einsum('mnae,mnie->ia',tIjaB,IjaB[ob,oa,oa,vb])
    #term 11
    _tia += (1/2)*np.einsum('imef,amef->ia',tijab,ijab[va,oa,va,va])
    #term 12
    _tia += (1/2)*np.einsum('imef,amef->ia',tiJaB,iJaB[va,ob,va,vb])
    #term 13
    _tia += (1/2)*np.einsum('imef,amef->ia',tiJAb,iJAb[va,ob,vb,va])
    _tia /= Dia
    return _tia

def update_tIA(fb,tia,tIA,tIJAB,tIjAb,tiJAb,tIjaB,IJAB,IjAb,iJAb,IjaB,DIA):
    # one particle intermediates
    FAE = form_FAE(fb,tia,tIA,tIJAB,tIjAb,tiJaB,IJAB,IjAb,iJAb)
    FMI = form_FMI(fb,tia,tIA,tIJAB,tIjAb,tIjaB,IJAB,IjAb,IjaB)
    Fme = form_Fme(fa,tia,tIA,ijab,iJaB)
    FME = form_FME(fb,tia,tIA,IJAB,IjAb)

    _tIA = np.zeros_like(tIA)
    #term 1
    _tIA += fb[ob,vb]
    #term 2
    _tIA += np.einsum('ie,ae->ia',tIA,FAE)
    #term 3
    _tIA -= np.einsum('ma,mi->ia',tIA,FMI)
    #term 4
    _tIA += np.einsum('imae,me->ia',tIJAB,FME)
    #term 5
    _tIA += np.einsum('imae,me->ia',tIjAb,Fme)
    #term 6
    _tIA += np.einsum('me,amie->ia',tIA,IJAB[vb,ob,ob,vb])
    #term 7
    _tIA += np.einsum('me,amie->ia',tia,IjAb[vb,oa,ob,va])
    #term 8
    _tIA -= (1/2)*np.einsum('mnae,mnie->ia',tIJAB,IJAB[ob,ob,ob,vb])
    #term 9
    _tIA -= (1/2)*np.einsum('mnae,mnie->ia',tIjAb,IjAb[ob,oa,ob,va])
    #term 10
    _tIA -= (1/2)*np.einsum('mnae,mnie->ia',tiJAb,iJAb[oa,ob,ob,va])
    #term 11
    _tIA += (1/2)*np.einsum('imef,amef->ia',tIJAB,IJAB[vb,ob,vb,vb])
    #term 12
    _tIA += (1/2)*np.einsum('imef,amef->ia',tIjAb,IjAb[vb,oa,vb,va])
    #term 13
    _tIA += (1/2)*np.einsum('imef,amef->ia',tIjaB,IjaB[vb,oa,va,vb])

    _tIA /= DIA

    return _tIA

def update_tijab(fa,tia,tIA,tijab,tiJaB,tIjaB,tiJAb,ijab,iJaB,IjaB,iJAb,IjAb):
    # one particle intermediates
    Fae = form_Fae(fa,tia,tIA,tijab,tiJaB,tIjaB,ijab,iJaB,IjaB)
    Fmi = form_Fmi(fa,tia,tIA,tijab,tiJaB,tiJAb,ijab,iJaB,iJAb)
    Fme = form_Fme(fa,tia,tIA,ijab,iJaB)
    # two particle intermediates
    Wmnij = form_Wmnij(tia,tijab,ijab)
    Wabef = form_Wabef(tia,tijab,ijab)
    Wmbej = form_Wmbej(tia,tijab,tiJAb,ijab,iJaB)
    WMbEj = form_WMbEj(tia,tijab,tiJAb,IjAb,IJAB)

    #expansion of (tia)(tia)
    tiatia = np.einsum('ma,nb->mnab',tia,tia)

    _tijab = np.zeros_like(ijab[oa,oa,va,va])
    #term 1
    _tijab += ijab[oa,oa,va,va]
    #term 2
    _tijab += np.einsum('ijae,be->ijab',tijab,Fae -\
              (1/2)*np.einsum('mb,me->be',tia,Fme))
    #term 3
    _tijab -= np.einsum('ijbe,ae->ijab',tijab,Fae -\
              (1/2)*np.einsum('ma,me->ae',tia,Fme))
    #term 4
    _tijab -= np.einsum('imab,mj->ijab',tijab,Fmi +\
              (1/2)*np.einsum('je,me->mj',tia,Fme))
    #term 5
    _tijab += np.einsum('jmab,mi->ijab',tijab,Fmi +\
              (1/2)*np.einsum('ie,me->mi',tia,Fme))
    #term 6
    _tijab += (1/2)*np.einsum('mnab,mnij->ijab',tijab + tiatia - tiatia.transpose(0,1,3,2),Wmnij)
    #term 7
    _tijab += (1/2)*np.einsum('ijef,abef->ijab',tijab + tiatia - tiatia.transpose(0,1,3,2),Wabef)
    #term 8a
    _tijab += np.einsum('imae,mbej->ijab',tijab,Wmbej)
    #term 8b
    _tijab -= np.einsum('imea,mbej->ijab',tiatia,ijab[oa,va,va,oa])
    #term 9
    _tijab += np.einsum('imae,mbej->ijab',tiJaB,WMbEj)
    #term 10a
    _tijab -= np.einsum('imbe,maej->ijab',tijab,Wmbej)
    #term 10b
    _tijab += np.einsum('imeb,maej->ijab',tiatia,ijab[oa,va,va,oa])
    #term 11
    _tijab -= np.einsum('imbe,maej->ijab',tiJaB,WMbEj)
    #term 12a
    _tijab -= np.einsum('jmae,mbei->ijab',tijab,Wmbej)
    #term 12b
    _tijab += np.einsum('jmea,mbei->ijab',tiatia,ijab[oa,va,va,oa])
    #term 13
    _tijab -= np.einsum('jmae,mbei->ijab',tiJaB,WMbEj)
    #term 14a
    _tijab += np.einsum('jmbe,maei->ijab',tijab,Wmbej)
    #term 14b
    _tijab -= np.einsum('jmeb,maei->ijab',tijab,ijab[oa,va,va,oa])
    #term 15
    _tijab += np.einsum('jmbe,maei->ijab',tiJaB,WMbEj)
    #term 16
    _tijab += np.einsum('ie,abej->ijab',tia,ijab[va,va,va,oa])
    #term 17
    _tijab -= np.einsum('je,abei->ijab',tia,ijab[va,va,va,oa])
    #term 18
    _tijab -= np.einsum('ma,mbij->ijab',tia,ijab[oa,va,oa,oa])
    #term 19
    _tijab += np.einsum('mb,maij->ijab',tia,ijab[oa,va,oa,oa])
    _tijab /= Dijab
    return _tijab

def update_tiJaB(tia,tIA,tiJaB,tiJAb,tIjaB,tIjAb,tIJAB,iJaB,IjaB,iJAb,IjAb):
    # one particle intermediates
    FAE = form_FAE(fb,tia,tIA,tIJAB,tIjAb,tiJaB,IJAB,IjAb,iJAb)
    Fae = form_Fae(fa,tia,tIA,tijab,tiJaB,tIjaB,ijab,iJaB,IjaB)
    FME = form_FME(fb,tia,tIA,IJAB,IjAb)
    Fme = form_Fme(fa,tia,tIA,ijab,iJaB)
    FMI = form_FMI(fb,tia,tIA,tIJAB,tIjAb,tIjaB,IJAB,IjAb,IjaB)
    Fmi = form_Fmi(fa,tia,tIA,tijab,tiJaB,tiJAb,ijab,iJaB,iJAb)
    # two particle intermediates
    WaBeF = form_WaBeF(tia,tIA,tiJaB,tIjaB,iJaB,IjaB)
    WaBEf = form_WaBEf(tia,tIA,tiJaB,tIjaB,iJAb,IjAb)
    WmNiJ = form_WmNiJ(tia,tIA,tiJaB,tiJAb,iJaB,iJAb)
    WMniJ = form_WMniJ(tia,tIA,tIjAb,tiJAb,IjaB,IjAb)
    WmBeJ = form_WmBeJ(tIA,tIJAB,tIjaB,iJaB,ijab)
    WMBEJ = form_WMBEJ(tIA,tIJAB,IJAB,IjAb)
    WMbeJ = form_WMbeJ(tia,tIA,tIjAb,IjaB)
    WmBEj = form_WmBEj(tia,tIA,tiJaB,iJAb)
    WMbEj = form_WMbEj(tia,tijab,tiJAb,IjAb,IJAB)
    Wmbej = form_Wmbej(tia,tijab,tiJAb,ijab,iJaB)
    #expansion of (tia)(tia)
    tiatia = np.einsum('ma,nb->mnab',tia,tia)
    #expansion of (tia)(tIA)
    tiatIA = np.einsum('ma,nb->mnab',tia,tIA)
    #expansion of (tIA)(tia)
    tIAtia = np.einsum('ma,nb->mnab',tIA,tia)
    #expansion of (tIA)(tIA)
    tIAtIA = np.einsum('ma,nb->mnab',tIA,tIA)

    _tiJaB = np.zeros_like(iJaB[oa,ob,va,vb])
    #term 1
    _tiJaB += iJaB[oa,ob,va,vb]
    #term 2
    _tiJaB += np.einsum('ijae,be->ijab',tiJaB, FAE -\
              (1/2)*np.einsum('mb,me->be',tIA,FME))
    #term 3
    _tiJaB -= np.einsum('ijbe,ae->ijab',tiJAb, Fae -\
              (1/2)*np.einsum('ma,me->ae',tia,Fme))
    #term 4
    _tiJaB -= np.einsum('imab,mj->ijab',tiJaB,FMI +\
              (1/2)*np.einsum('je,me->mj',tIA,FME))
    #term 5
    _tiJaB += np.einsum('jmab,mi->ijab',tIjaB,Fmi +\
              (1/2)*np.einsum('ie,me->mi',tia,Fme))
    #term 6
    _tiJaB += (1/2)*np.einsum('mnab,mnij->ijab',tiJaB + tiatIA,WmNiJ)
    #term 7
    _tiJaB += (1/2)*np.einsum('mnab,mnij->ijab',tIjaB - tIAtia,WMniJ)
    #term 8
    _tiJaB += (1/2)*np.einsum('ijef,abef->ijab',tiJaB + tiatIA,WaBeF)
    #term 9
    _tiJaB += (1/2)*np.einsum('ijef,abef->ijab',tiJAb - tiatIA,WaBEf)
    #term 10a
    _tiJaB += np.einsum('imae,mbej->ijab',tijab,WmBeJ)
    #term 10b
    _tiJaB -= np.einsum('imea,mbej->ijab',tijab,iJaB[oa,vb,va,ob])
    #term 11
    _tiJaB += np.einsum('imae,mbej->ijab',tiJaB,WMBEJ)
    #term 12a
    _tiJaB -= np.einsum('imbe,maej->ijab',tiJAb,WMbeJ)
    #term 12b
    _tiJaB += np.einsum('imeb,maej->ijab',tiJaB,IjaB[ob,va,va,ob])
    #term 13a
    _tiJaB -= np.einsum('jmae,mbei->ijab',tIjaB,WmBEj)
    #term 13b
    _tiJaB += np.einsum('jmea,mbei->ijab',tIjAb,iJAb[oa,vb,vb,oa])
    #term 14a
    _tiJaB += np.einsum('jmbe,maei->ijab',tIJAB,WMbEj)
    #term 14b
    _tiJaB -= np.einsum('jmeb,maei->ijab',tIJAB,IjAb[ob,va,vb,oa])
    #term 15
    _tiJaB += np.einsum('jmbe,maei->ijab',tIjAb,Wmbej)
    #term 16
    _tiJaB += np.einsum('ie,abej->ijab',tia,iJaB[va,vb,va,ob])
    #term 17
    _tiJaB -= np.einsum('je,abei->ijab',tIA,iJAb[va,vb,vb,oa])
    #term 18
    _tiJaB -= np.einsum('ma,mbij->ijab',tia,iJaB[oa,vb,oa,ob])
    #term 19
    _tiJaB += np.einsum('mb,maij->ijab',tIA,IjaB[ob,va,oa,ob])
    _tiJaB /= DiJaB
    return _tiJaB

def update_tiJAb(tia,tIA,tiJAb,tiJaB,tIjAb,tijab,tIJAB,tIjaB,IjaB,iJaB,IjAb):
    # one particle intermediates
    FAE = form_FAE(fb,tia,tIA,tIJAB,tIjAb,tiJaB,IJAB,IjAb,iJAb)
    Fae = form_Fae(fa,tia,tIA,tijab,tiJaB,tIjaB,ijab,iJaB,IjaB)
    FME = form_FME(fb,tia,tIA,IJAB,IjAb)
    Fme = form_Fme(fa,tia,tIA,ijab,iJaB)
    FMI = form_FMI(fb,tia,tIA,tIJAB,tIjAb,tIjaB,IJAB,IjAb,IjaB)
    Fmi = form_Fmi(fa,tia,tIA,tijab,tiJaB,tiJAb,ijab,iJaB,iJAb)
    # two particle intermediates
    WAbeF = form_WAbeF(tia,tIA,tIjAb,tiJAb,IjaB,iJaB)
    WAbEf = form_WAbEf(tia,tIA,tIjAb,tiJAb,IjAb,iJAb)
    WmNiJ = form_WmNiJ(tia,tIA,tiJaB,tiJAb,iJaB,iJAb)
    WMniJ = form_WMniJ(tia,tIA,tIjAb,tiJAb,IjaB,IjAb)
    WmBeJ = form_WmBeJ(tIA,tIJAB,tIjaB,iJaB,ijab)
    WMBEJ = form_WMBEJ(tIA,tIJAB,IJAB,IjAb)
    WMbeJ = form_WMbeJ(tia,tIA,tIjAb,IjaB)
    WmBEj = form_WmBEj(tia,tIA,tiJaB,iJAb)
    WMbEj = form_WMbEj(tia,tijab,tiJAb,IjAb,IJAB)
    Wmbej = form_Wmbej(tia,tijab,tiJAb,ijab,iJaB)
    #expansion of (tia)(tia)
    tiatia = np.einsum('ma,nb->mnab',tia,tia)
    #expansion of (tia)(tIA)
    tiatIA = np.einsum('ma,nb->mnab',tia,tIA)
    #expansion of (tIA)(tia)
    tIAtia = np.einsum('ma,nb->mnab',tIA,tia)
    #expansion of (tIA)(tIA)
    tIAtIA = np.einsum('ma,nb->mnab',tIA,tIA)

    _tiJAb = np.zeros_like(tiJaB)
    #term 1
    _tiJAb += iJAb[oa,ob,vb,va]
    #term 2
    _tiJAb += np.einsum('ijae,be->ijab',tiJAb,Fae -\
              (1/2)*np.einsum('mb,me->be',tia,Fme))
    #term 3
    _tiJAb -= np.einsum('ijbe,ae->ijab',tiJaB,FAE -\
              (1/2)*np.einsum('ma,me->ae',tIA,FME))
    #term 4
    _tiJAb -= np.einsum('imab,mj->ijab',tiJAb,FMI +\
              (1/2)*np.einsum('je,me->mj',tIA,FME))
    #term 5
    _tiJAb += np.einsum('jmab,mi->ijab',tIjAb,Fmi +\
              (1/2)*np.einsum('ie,me->mi',tia,Fme))
    #term 6
    _tiJAb += (1/2)*np.einsum('mnab,mnij->ijab',tIjAb + tIAtia,WMniJ)
    #term 7
    _tiJAb += (1/2)*np.einsum('mnab,mnij->ijab',tiJAb - tiatIA,WmNiJ)
    #term 8
    _tiJAb += (1/2)*np.einsum('ijef,abef->ijab',tiJaB + tiatIA,WAbeF)
    #term 9
    _tiJAb += (1/2)*np.einsum('ijef,abef->ijab',tiJAb - tiatIA,WAbEf)
    #term 10a
    _tiJAb += np.einsum('imae,mbej->ijab',tiJAb,WMbeJ)
    #term 10b
    _tiJAb -= np.einsum('imea,mbej->ijab',tiatIA,IjaB[ob,va,va,ob])
    #term 11a
    _tiJAb -= np.einsum('imbe,maej->ijab',tijab,WmBeJ)
    #term 11b
    _tiJAb += np.einsum('imeb,maej->ijab',tiatIA,iJaB[oa,vb,va,ob])
    #term 12
    _tiJAb -= np.einsum('imbe,maej->ijab',tiJaB,WMBEJ)
    #term 13a
    _tiJAb -= np.einsum('jmae,mbei->ijab',tIJAB,WMbEj)
    #term 13b
    _tiJAb += np.einsum('jmea,mbei->ijab',tIAtia,IjAb[ob,va,vb,oa])
    #term 14
    _tiJAb -= np.einsum('jmae,mbei->ijab',tIjAb,Wmbej)
    #term 15a
    _tiJAb += np.einsum('jmbe,maei->ijab',tIjaB,WmBEj)
    #term 15b
    _tiJAb -= np.einsum('jmeb,maei->ijab',tIAtia,iJAb[oa,vb,vb,oa])
    #term 16
    _tiJAb += np.einsum('ie,abej->ijab',tia,IjaB[vb,va,va,ob])
    #term 17
    _tiJAb -= np.einsum('je,abei->ijab',tIA,IjAb[vb,va,vb,oa])
    #term 18
    _tiJAb -= np.einsum('ma,mbij->ijab',tIA,IjaB[ob,va,oa,ob])
    #term 19
    _tiJAb += np.einsum('mb,maij->ijab',tia,iJaB[oa,vb,oa,ob])
    _tiJAb /= DiJAb
    return _tiJAb

def update_tIJAB(fb,tia,tIA,tIJAB,tIjAb,tiJaB,tIjaB,IJAB,ijab,iJaB,IjAb):
    # one particle intermediates
    FAE = form_FAE(fb,tia,tIA,tIJAB,tIjAb,tiJaB,IJAB,IjAb,iJAb)
    FME = form_FME(fb,tia,tIA,IJAB,IjAb)
    FMI = form_FMI(fb,tia,tIA,tIJAB,tIjAb,tIjaB,IJAB,IjAb,IjaB)
    # two particle intermediates
    WMNIJ = form_WMNIJ(tIA,tIJAB,IJAB)
    WABEF = form_WABEF(tIA,tIJAB,IJAB)
    WMBEJ = form_WMBEJ(tIA,tIJAB,IJAB,IjAb)
    WmBeJ = form_WmBeJ(tIA,tIJAB,tIjaB,iJaB,ijab)
    #expansion of (tIA)(tIA)
    tIAtIA = np.einsum('ma,nb->mnab',tIA,tIA)

    _tIJAB = np.zeros_like(IJAB[ob,ob,vb,vb])
    #term 1
    _tIJAB += IJAB[ob,ob,vb,vb]
    #term 2
    _tIJAB += np.einsum('ijae,be->ijab',tIJAB,FAE -\
              (1/2)*np.einsum('mb,me->be',tIA,FME))
    #term 3
    _tIJAB -= np.einsum('ijbe,ae->ijab',tIJAB,FAE -\
              (1/2)*np.einsum('ma,me->ae',tIA,FME))
    #term 4
    _tIJAB -= np.einsum('imab,mj->ijab',tIJAB,FMI +\
              (1/2)*np.einsum('je,me->mj',tIA,FME))
    #term 5
    _tIJAB += np.einsum('jmab,mi->ijab',tIJAB,FMI +\
              (1/2)*np.einsum('ie,me->mi',tIA,FME))
    #term 6
    _tIJAB += (1/2)*np.einsum('mnab,mnij->ijab',tIJAB + tIAtIA - tIAtIA.transpose(0,1,3,2), WMNIJ)
    #term 7
    _tIJAB += (1/2)*np.einsum('ijef,abef->ijab',tIJAB + tIAtIA - tIAtIA.transpose(0,1,3,2), WABEF)
    #term 8a
    _tIJAB += np.einsum('imae,mbej->ijab',tIJAB,WMBEJ)
    #term 8b
    _tIJAB -= np.einsum('imea,mbej->ijab',tIAtIA,IJAB[ob,vb,vb,ob])
    #term 9
    _tIJAB += np.einsum('imae,mbej->ijab',tIjAb,WmBeJ)
    #term 10a
    _tIJAB -= np.einsum('imbe,maej->ijab',tIJAB,WMBEJ)
    #term 10b
    _tIJAB += np.einsum('imeb,maej->ijab',tIAtIA,IJAB[ob,vb,vb,ob])
    #term 11
    _tIJAB -= np.einsum('imbe,maej->ijab',tIjAb,WmBeJ)
    #term 12a
    _tIJAB -= np.einsum('jmae,mbei->ijab',tIJAB,WMBEJ)
    #term 12b
    _tIJAB += np.einsum('jmea,mbei->ijab',tIAtIA,IJAB[ob,vb,vb,ob])
    #term 13
    _tIJAB -= np.einsum('jmae,mbei->ijab',tIjAb,WmBeJ)
    #term 14a
    _tIJAB += np.einsum('jmbe,maei->ijab',tIJAB,WMBEJ)
    #term 14b
    _tIJAB -= np.einsum('jmeb,maei->ijab',tIAtIA,IJAB[ob,vb,vb,ob])
    #term 15
    _tIJAB += np.einsum('jmbe,maei->ijab',tIjAb,WmBeJ)
    #term 16
    _tIJAB += np.einsum('ie,abej->ijab',tIA,IJAB[vb,vb,vb,ob])
    #term 17
    _tIJAB -= np.einsum('je,abei->ijab',tIA,IJAB[vb,vb,vb,ob])
    #term 18
    _tIJAB -= np.einsum('ma,mbij->ijab',tIA,IJAB[ob,vb,ob,ob])
    #term 19
    _tIJAB += np.einsum('mb,maij->ijab',tIA,IJAB[ob,vb,ob,ob])
    _tIJAB /= DIJAB
    return _tIJAB

def update_tIjAb(tia,tIA,tIjAb,tIjaB,tiJAb,tijab,iJAb,IjaB,iJaB,ijab,IJAB):
    # one particle intermediates
    FAE = form_FAE(fb,tia,tIA,tIJAB,tIjAb,tiJaB,IJAB,IjAb,iJAb)
    Fae = form_Fae(fa,tia,tIA,tijab,tiJaB,tIjaB,ijab,iJaB,IjaB)
    FME = form_FME(fb,tia,tIA,IJAB,IjAb)
    Fme = form_Fme(fa,tia,tIA,ijab,iJaB)
    FMI = form_FMI(fb,tia,tIA,tIJAB,tIjAb,tIjaB,IJAB,IjAb,IjaB)
    Fmi = form_Fmi(fa,tia,tIA,tijab,tiJaB,tiJAb,ijab,iJaB,iJAb)
    # two particle intermediates
    WAbeF = form_WAbeF(tia,tIA,tIjAb,tiJAb,IjaB,iJaB)
    WAbEf = form_WAbEf(tia,tIA,tIjAb,tiJAb,IjAb,iJAb)
    WMnIj = form_WMnIj(tia,tIA,tIjAb,tIjaB,IjAb,IjaB)
    WmNIj = form_WmNIj(tia,tIA,tIjAb,tIjaB,iJAb,iJaB)
    WmBeJ = form_WmBeJ(tIA,tIJAB,tIjaB,iJaB,ijab)
    WMBEJ = form_WMBEJ(tIA,tIJAB,IJAB,IjAb)
    WMbeJ = form_WMbeJ(tia,tIA,tIjAb,IjaB)
    WmBEj = form_WmBEj(tia,tIA,tiJaB,iJAb)
    WMbEj = form_WMbEj(tia,tijab,tiJAb,IjAb,IJAB)
    Wmbej = form_Wmbej(tia,tijab,tiJAb,ijab,iJaB)
    #expansion of (tia)(tia)
    tiatia = np.einsum('ma,nb->mnab',tia,tia)
    #expansion of (tia)(tIA)
    tiatIA = np.einsum('ma,nb->mnab',tia,tIA)
    #expansion of (tIA)(tia)
    tIAtia = np.einsum('ma,nb->mnab',tIA,tia)
    #expansion of (tIA)(tIA)
    tIAtIA = np.einsum('ma,nb->mnab',tIA,tIA)
    
    _tIjAb = np.zeros_like(IjAb[ob,oa,vb,va])
    #term 1
    _tIjAb += IjAb[ob,oa,vb,va]
    #term 2
    _tIjAb += np.einsum('ijae,be->ijab',tIjAb,Fae -\
             (1/2)*np.einsum('mb,me->be',tia,Fme))
    #term 3
    _tIjAb -= np.einsum('ijbe,ae->ijab',tIjaB,FAE -\
             (1/2)*np.einsum('ma,me->ae',tIA,FME))
    #term 4
    _tIjAb -= np.einsum('imab,mj->ijab',tIjAb,Fmi +\
             (1/2)*np.einsum('je,me->mj',tia,Fme))
    #term 5
    _tIjAb += np.einsum('jmab,mi->ijab',tiJAb,FMI +\
             (1/2)*np.einsum('ie,me->mi',tIA,FME))
    #term 6
    _tIjAb += (1/2)*np.einsum('mnab,mnij->ijab',tIjAb + tIAtia, WMnIj)
    #term 7
    _tIjAb += (1/2)*np.einsum('mnab,mnij->ijab',tiJAb - tiatIA, WmNIj)
    #term 8
    _tIjAb += (1/2)*np.einsum('ijef,abef->ijab',tIjAb + tIAtia, WAbEf)
    #term 9
    _tIjAb += (1/2)*np.einsum('ijef,abef->ijab',tIjaB - tIAtia, WAbeF)
    #term 10a
    _tIjAb += np.einsum('imae,mbej->ijab',tIJAB,WMbEj)
    #term 10b
    _tIjAb -= np.einsum('imea,mbej->ijab',tIAtIA,IjAb[ob,va,vb,oa])
    #term 11
    _tIjAb += np.einsum('imae,mbej->ijab',tIjAb,Wmbej)
    #term 12a
    _tIjAb -= np.einsum('imbe,maej->ijab',tIjaB,WmBEj)
    #term 12b
    _tIjAb += np.einsum('imeb,maej->ijab',tIAtia,iJAb[oa,vb,vb,oa])
    #term 13a
    _tIjAb -= np.einsum('jmae,mbei->ijab',tiJAb,WMbeJ)
    #term 13b
    _tIjAb += np.einsum('jmea,mbei->ijab',tiatIA,IjaB[ob,va,va,ob])
    #term 14a
    _tIjAb += np.einsum('jmbe,maei->ijab',tijab,WmBeJ)
    #term 14b
    _tIjAb -= np.einsum('jmeb,maei->ijab',tiatIA,iJaB[oa,vb,va,ob])
    #term 15
    _tIjAb += np.einsum('jmbe,maei->ijab',tiJaB,WMBEJ)
    #term 16
    _tIjAb += np.einsum('ie,abej->ijab',tIA,IjAb[vb,va,vb,oa])
    #term 17
    _tIjAb -= np.einsum('je,abei->ijab',tia,IjaB[vb,va,va,ob])
    #term 18
    _tIjAb -= np.einsum('ma,mbij->ijab',tIA,IjAb[ob,va,ob,oa])
    #term 19
    _tIjAb += np.einsum('mb,maij->ijab',tia,iJAb[oa,vb,ob,oa])
    _tIjAb /= DIjAb
    return _tIjAb

def update_tIjaB(tia,tIA,tIjaB,tIjAb,tiJaB,tIJAB,tijab,tiJAb,IjaB,iJAb,IjAb,iJaB):
    # one particle intermediates
    FAE = form_FAE(fb,tia,tIA,tIJAB,tIjAb,tiJaB,IJAB,IjAb,iJAb)
    Fae = form_Fae(fa,tia,tIA,tijab,tiJaB,tIjaB,ijab,iJaB,IjaB)
    FME = form_FME(fb,tia,tIA,IJAB,IjAb)
    Fme = form_Fme(fa,tia,tIA,ijab,iJaB)
    FMI = form_FMI(fb,tia,tIA,tIJAB,tIjAb,tIjaB,IJAB,IjAb,IjaB)
    Fmi = form_Fmi(fa,tia,tIA,tijab,tiJaB,tiJAb,ijab,iJaB,iJAb)
    # two particle intermediates
    WmNIj = form_WmNIj(tia,tIA,tIjAb,tIjaB,iJAb,iJaB)
    WMnIj = form_WMnIj(tia,tIA,tIjAb,tIjaB,IjAb,IjaB)
    WaBEf = form_WaBEf(tia,tIA,tiJaB,tIjaB,iJAb,IjAb)
    WaBeF = form_WaBeF(tia,tIA,tiJaB,tIjaB,iJaB,IjaB)
    WmBEj = form_WmBEj(tia,tIA,tiJaB,iJAb)
    WMbEj = form_WMbEj(tia,tijab,tiJAb,IjAb,IJAB)
    Wmbej = form_Wmbej(tia,tijab,tiJAb,ijab,iJaB)
    WmBeJ = form_WmBeJ(tIA,tIJAB,tIjaB,iJaB,ijab)
    WMBEJ = form_WMBEJ(tIA,tIJAB,IJAB,IjAb)
    WMbeJ = form_WMbeJ(tia,tIA,tIjAb,IjaB)
    #expansion of (tia)(tia)
    tiatia = np.einsum('ma,nb->mnab',tia,tia)
    #expansion of (tia)(tIA)
    tiatIA = np.einsum('ma,nb->mnab',tia,tIA)
    #expansion of (tIA)(tia)
    tIAtia = np.einsum('ma,nb->mnab',tIA,tia)
    #expansion of (tIA)(tIA)
    tIAtIA = np.einsum('ma,nb->mnab',tIA,tIA)

    _tIjaB = np.zeros_like(tIjaB)
    #term 1
    _tIjaB += IjaB[ob,oa,va,vb]
    #term 2
    _tIjaB += np.einsum('ijae,be->ijab',tIjaB,FAE -\
              (1/2)*np.einsum('mb,me->be',tIA,FME))
    #term 3
    _tIjaB -= np.einsum('ijbe,ae->ijab',tIjAb,Fae -\
              (1/2)*np.einsum('ma,me->ae',tia,Fme))
    #term 4
    _tIjaB -= np.einsum('imab,mj->ijab',tIjaB,Fmi +\
              (1/2)*np.einsum('je,me->mj',tia,Fme))
    #term 5
    _tIjaB += np.einsum('jmab,mi->ijab',tiJaB,FMI +\
              (1/2)*np.einsum('ie,me->mi',tIA,FME))
    #term 6
    _tIjaB += (1/2)*np.einsum('mnab,mnij->ijab',tiJaB + tiatIA,WmNIj)
    #term 7
    _tIjaB += (1/2)*np.einsum('mnab,mnij->ijab',tIjaB - tIAtia,WMnIj)
    #term 8
    _tIjaB += (1/2)*np.einsum('ijef,abef->ijab',tIjAb + tIAtia,WaBEf)
    #term 9
    _tIjaB += (1/2)*np.einsum('ijef,abef->ijab',tIjaB - tIAtia,WaBeF)
    #term 10a
    _tIjaB += np.einsum('imae,mbej->ijab',tIjaB,WmBEj)
    #term 10b
    _tIjaB -= np.einsum('imea,mbej->ijab',tIjAb,iJAb[oa,vb,vb,oa])
    #term 11a
    _tIjaB -= np.einsum('imbe,maej->ijab',tIJAB,WMbEj)
    #term 11b
    _tIjaB += np.einsum('imeb,maej->ijab',tIAtIA,IjAb[ob,va,vb,oa])
    #term 12
    _tIjaB -= np.einsum('imbe,maej->ijab',tIjAb,Wmbej)
    #term 13a
    _tIjaB -= np.einsum('jmae,mbei->ijab',tijab,WmBeJ)
    #term 13b
    _tIjaB += np.einsum('jmea,mbei->ijab',tiatia,iJaB[oa,vb,va,ob])
    #term 14
    _tIjaB -= np.einsum('jmae,mbei->ijab',tiJaB,WMBEJ)
    #term 15a
    _tIjaB += np.einsum('jmbe,maei->ijab',tiJAb,WMbeJ)
    #term 15b
    _tIjaB -= np.einsum('jmeb,maei->ijab',tiatIA,IjaB[ob,va,va,ob])
    #term 16
    _tIjaB += np.einsum('ie,abej->ijab',tIA,iJAb[va,vb,vb,oa])
    #term 17
    _tIjaB -= np.einsum('je,abei->ijab',tia,iJaB[va,vb,va,ob])
    #term 18
    _tIjaB -= np.einsum('ma,mbij->ijab',tia,iJAb[oa,vb,ob,oa])
    #term 19
    _tIjaB += np.einsum('mb,maij->ijab',tIA,IjAb[ob,va,ob,oa])
    _tIjaB /= DIjaB
    return _tIjaB

Wmnij = form_Wmnij(tia,tijab,ijab)
WmNiJ = form_WmNiJ(tia,tIA,tiJaB,tiJAb,iJaB,iJAb)
WmNIj = form_WmNIj(tia,tIA,tIjAb,tIjaB,iJAb,iJaB)
WMNIJ = form_WMNIJ(tIA,tIJAB,IJAB)
WMnIj = form_WMnIj(tia,tIA,tIjAb,tIjaB,IjAb,IjaB)
WMniJ = form_WMniJ(tia,tIA,tIjAb,tiJAb,IjaB,IjAb)

Wabef = form_Wabef(tia,tijab,ijab)
WaBeF = form_WaBeF(tia,tIA,tiJaB,tIjaB,iJaB,IjaB)
WaBEf = form_WaBEf(tia,tIA,tiJaB,tIjaB,iJAb,IjAb)
WABEF = form_WABEF(tIA,tIJAB,IJAB)
WAbEf = form_WAbEf(tia,tIA,tIjAb,tiJAb,IjAb,iJAb)
WAbeF = form_WAbeF(tia,tIA,tIjAb,tiJAb,IjaB,iJaB)

Wmbej = form_Wmbej(tia,tijab,tiJAb,ijab,iJaB)
WmBeJ = form_WmBeJ(tIA,tIJAB,tIjaB,iJaB,ijab)
WmBEj = form_WmBEj(tia,tIA,tiJaB,iJAb)
WMBEJ = form_WMBEJ(tIA,tIJAB,IJAB,IjAb)
WMbEj = form_WMbEj(tia,tijab,tiJAb,IjAb,IJAB)
WMbeJ = form_WMbeJ(tia,tIA,tIjAb,IjaB)

tia_new = update_tia(fa,tia,tIA,tijab,tiJaB,tIjaB,tiJAb,ijab,iJaB,IjaB,iJAb,Dia)
tIA_new = update_tIA(fb,tia,tIA,tIJAB,tIjAb,tiJAb,tIjaB,IJAB,IjAb,iJAb,IjaB,DIA)

tijab_new = update_tijab(fa,tia,tIA,tijab,tiJaB,tIjaB,tiJAb,ijab,iJaB,IjaB,iJAb,IjAb)
tiJaB_new = update_tiJaB(tia,tIA,tiJaB,tiJAb,tIjaB,tIjAb,tIJAB,iJaB,IjaB,iJAb,IjAb)
tiJAb_new = update_tiJAb(tia,tIA,tiJAb,tiJaB,tIjAb,tijab,tIJAB,tIjaB,IjaB,iJaB,IjAb)
tIJAB_new = update_tIJAB(fb,tia,tIA,tIJAB,tIjAb,tiJaB,tIjaB,IJAB,ijab,iJaB,IjAb)
tIjAb_new = update_tIjAb(tia,tIA,tIjAb,tIjaB,tiJAb,tijab,iJAb,IjaB,iJaB,ijab,IJAB)
tIjaB_new = update_tIjaB(tia,tIA,tIjaB,tIjAb,tiJaB,tIJAB,tijab,tiJAb,IjaB,iJAb,IjAb,iJaB)
