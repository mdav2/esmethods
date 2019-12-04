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

print("E[SCF]: ", escf + enuc)
print("E[SCF/Psi4]: ", e)

#form fock matrix in MO basis
f =  np.zeros_like(h)
f += h
f += 2*np.einsum('pqkk->pq',mo_eri.np[:,:,:ndocc,:ndocc])
f -= np.einsum('pkqk->pq',mo_eri.np[:,:ndocc,:,:ndocc])

#form antisymmetrized MO integrals
ijab = np.zeros_like(mo_eri.np)
iJaB = np.zeros_like(mo_eri.np)
iJAb = np.zeros_like(mo_eri.np)
for i in range(nbf):
    for j in range(nbf):
        for a in range(nbf):
            for b in range(nbf):
                ijab[i][j][a][b] =  mo_eri.np[i][a][j][b] - mo_eri.np[i][b][j][a]
                iJaB[i][j][a][b] =  mo_eri.np[i][a][j][b]
                iJAb[i][j][a][b] = -mo_eri.np[i][b][j][a]


#form denominator array
#just one spin case
Dia = np.zeros((ndocc,nbf-ndocc))
for i in range(ndocc):
    for a in range(nbf-ndocc):
        aa = a + ndocc
        Dia[i][a] = f[i][i] - f[aa][aa]

Dijab = np.zeros((ndocc,ndocc,nbf-ndocc,nbf-ndocc))
for i in range(ndocc):
    for j in range(ndocc):
        for a in range(nbf-ndocc):
            for b in range(nbf-ndocc):
                aa = a + ndocc
                bb = b + ndocc
                Dijab[i][j][a][b] = f[i][i] + f[j][j]  - f[aa][aa] - f[bb][bb]
#form initial T1
tia = f[:ndocc,ndocc:]/Dia

#form initial T2
tijab = ijab[:ndocc,:ndocc,ndocc:,ndocc:]/Dijab
tiJaB = iJaB[:ndocc,:ndocc,ndocc:,ndocc:]/Dijab
tiJAb = iJAb[:ndocc,:ndocc,ndocc:,ndocc:]/Dijab

emp2 = 0.0
emp2 = np.einsum('ijab,ijab->',ijab[:ndocc,:ndocc,ndocc:,ndocc:],tijab)/2
emp2 += np.einsum('ijab,ijab->',iJaB[:ndocc,:ndocc,ndocc:,ndocc:],tiJaB)/2
emp2 += np.einsum('ijab,ijab->',iJAb[:ndocc,:ndocc,ndocc:,ndocc:],tiJAb)/2
print(emp2)

def form_Fae(tia,iJaB,tiJaB):
    Fae  =  np.einsum('mf,amef->ae',tia,2*iJaB[ndocc:,:ndocc,ndocc:,ndocc:])
    Fae -= np.einsum('mf,maef->ae',tia,iJaB[:ndocc,ndocc:,ndocc:,ndocc:])
    temp_1 = tiJaB + np.einsum('ma,nf->mnaf',tia,tia)/2
    temp_2 = 2*iJaB - iJaB.transpose((1,0,2,3))
    temp_2 = temp_2[:ndocc,:ndocc,ndocc:,ndocc:]
    Fae -= np.einsum('mnaf,mnef->ae',temp_1,temp_2)
    return Fae

def form_Fmi(tia,iJaB,tiJaB,tijab):
    Fmi  =   np.einsum('ne,mnie->mi',tia,2*iJaB[:ndocc,:ndocc,:ndocc,ndocc:])
    Fmi -=   np.einsum('ne,nmie->mi',tia,  iJaB[:ndocc,:ndocc,:ndocc,ndocc:])
    Fmi +=   np.einsum('inef,mnef->mi',tiJaB,2*iJaB[:ndocc,:ndocc,ndocc:,ndocc:])
    Fmi -=   np.einsum('inef,mnfe->mi',tiJaB,iJaB[:ndocc,:ndocc,ndocc:,ndocc:])
    temp_1 = np.einsum('ie,nf->inef',tia,tia)/2
    Fmi +=   np.einsum('inef,mnef->mi',temp_1,2*iJaB[:ndocc,:ndocc,ndocc:,ndocc:])
    Fmi -=   np.einsum('inef,mnfe->mi',temp_1,iJaB[:ndocc,:ndocc,ndocc:,ndocc:])
    return Fmi

def form_Fme(tia,iJaB):
    Fme  =  np.einsum('nf,mnef->me',tia,2*iJaB[:ndocc,:ndocc,ndocc:,ndocc:])
    Fme -= np.einsum('nf,nmef->me',tia,iJaB[:ndocc,:ndocc,ndocc:,ndocc:])
    return Fme

def form_Wmnij(iJaB,tia,tiJaB):
    Wmnij = np.zeros_like(iJaB[:ndocc,:ndocc,:ndocc,:ndocc])
    Wmnij += iJaB[:ndocc,:ndocc,:ndocc,:ndocc]
    Wmnij += np.einsum('je,mnie->mnij',tia,iJaB[:ndocc,:ndocc,:ndocc,ndocc:])
    Wmnij += np.einsum('ie,mnej->mnij',tia,iJaB[:ndocc,:ndocc,ndocc:,:ndocc])
    Wmnij += np.einsum('ijef,mnef->mnij',tiJaB,iJaB[:ndocc,:ndocc,ndocc:,ndocc:])/2
    temp_1 = np.einsum('ie,jf->ijef',tia,tia)
    Wmnij += np.einsum('ijef,mnef->mnij',temp_1,iJaB[:ndocc,:ndocc,ndocc:,ndocc:])/2
    return Wmnij

def form_Wabef( iJaB,tia,tiJaB):
    Wabef = np.zeros_like(iJaB[ndocc:,ndocc:,ndocc:,ndocc:])
    Wabef += iJaB[ndocc:,ndocc:,ndocc:,ndocc:]
    Wabef -= np.einsum('mb,amef->abef',tia,iJaB[ndocc:,:ndocc,ndocc:,ndocc:])
    Wabef -= np.einsum('ma,mbef->abef',tia,iJaB[:ndocc,ndocc:,ndocc:,ndocc:])
    Wabef += np.einsum('mnab,mnef->abef',tiJaB,iJaB[:ndocc,:ndocc,ndocc:,ndocc:])/2
    temp_1 = np.einsum('ma,nb->mnab',tia,tia)
    Wabef += np.einsum('mnab,mnef',temp_1,iJaB[:ndocc,:ndocc,ndocc:,ndocc:])/2
    return Wabef

def form_WmBeJ(iJaB,tia,tiJaB):
    WmBeJ = np.zeros_like(iJaB[:ndocc,ndocc:,ndocc:,:ndocc])
    WmBeJ += iJaB[:ndocc,ndocc:,ndocc:,:ndocc]
    WmBeJ += np.einsum('jf,mbef->mbej',tia,iJaB[:ndocc,ndocc:,ndocc:,ndocc:])
    WmBeJ -= np.einsum('nb,mnej->mbej',tia,iJaB[:ndocc,:ndocc,ndocc:,:ndocc])
    WmBeJ -= np.einsum('jnfb,mnef->mbej',tiJaB,iJaB[:ndocc,:ndocc,ndocc:,ndocc:])/2
    temp_1 = np.einsum('jf,nb->jnfb',tia,tia)
    WmBeJ -= np.einsum('jnfb,mnef->mbej',2*temp_1,iJaB[:ndocc,:ndocc,ndocc:,ndocc:])/2
    WmBeJ += np.einsum('njfb,mnef->mbej',tiJaB,2*iJaB[:ndocc,:ndocc,ndocc:,ndocc:])/2
    WmBeJ -= np.einsum('njfb,nmef->mbej',tiJaB,iJaB[:ndocc,:ndocc,ndocc:,ndocc:])/2
    return WmBeJ

def form_WmBEj(iJaB,tia,tiJaB):
    WmBEj = np.zeros_like(iJaB[:ndocc,ndocc:,ndocc:,:ndocc])
    WmBEj += -iJaB[:ndocc,ndocc:,ndocc:,:ndocc]
    WmBEj -= np.einsum('jf,mbfe->mbej',tia,iJaB[:ndocc,ndocc:,ndocc:,ndocc:])
    WmBEj += np.einsum('nb,nmej->mbej',tia,iJaB[:ndocc,:ndocc,ndocc:,:ndocc])
    WmBEj += np.einsum('jnfb,nmef->mbej',tiJaB,iJaB[:ndocc,:ndocc,ndocc:,ndocc:])/2
    temp_1 = np.einsum('jf,nb->jnfb',tia,tia)
    WmBEj += np.einsum('jnfb,nmef->mbej',temp_1,iJaB[:ndocc,:ndocc,ndocc:,ndocc:])
    return WmBEj

def update_T1(tia,Fae,tiJaB,iJaB):
    assert tia.shape == Fae.shape
    _tia = np.zeros_like(tia)
    _tia += np.einsum('ie,ae->ia',tia,Fae)
    _tia -= np.einsum('ma,mi->ia',tia,Fae)
#left off: these functions run w/o errors.
#need to code up T1 & T2 update, ccenergy.
Fae   = form_Fae(tia,iJaB,tiJaB)
Fmi   = form_Fmi(tia,iJaB,tiJaB,tijab)
Fme   = form_Fme(tia,iJaB)
Wmnij = form_Wmnij(iJaB,tia,tiJaB)
Wabef = form_Wabef(iJaB,tia,tiJaB)
WmBeJ = form_WmBeJ(iJaB,tia,tiJaB)
WmBEj = form_WmBEj(iJaB,tia,tiJaB)
tia   = update_T1(tia,Fae,)