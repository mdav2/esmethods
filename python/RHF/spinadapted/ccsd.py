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
        O
        H 1 1.1
        H 1 1.1 2 104.0
        symmetry c1
        """)
enuc = mol.nuclear_repulsion_energy()

psi4.set_options({'basis':'cc-pvtz',
                  'e_convergence':1e-12,
                  'd_convergence':1e-10,
                  'scf_type':'pk'})
disk_T2   = True
disk_iJaB = True
prec = 'float32' #increase to float64 if you need bunches of precision

e,wfn = psi4.energy('scf',mol=mol,return_wfn=True)
mints = psi4.core.MintsHelper(wfn.basisset())

ea = wfn.epsilon_a().to_array()
nbf = len(ea)
_Ca = wfn.Ca()
Ca = _Ca.to_array()
mo_eri = mints.mo_eri(_Ca,_Ca,_Ca,_Ca)
_h = wfn.H().to_array()
ndocc = wfn.nalpha()
o = slice(0,ndocc)
v = slice(ndocc,nbf)


h = np.einsum('mq,np,mn->pq',Ca,Ca,_h)

escf = 2*np.einsum('ii->',h[:ndocc,:ndocc])

#for i in range(ndocc):
#    for j in range(ndocc):
#        escf += 2*mo_eri.np[i][i][j][j] - mo_eri.np[i][j][i][j]
# ^                          ^  ^  ^  ^ 
# +----- this for loop ...   |  |  |  |
#                            +--+  +--+ 
#                            
# +----- ... and this einsum do the same thing
# V
escf += 2*np.einsum('iijj->',mo_eri.np[:ndocc,:ndocc,:ndocc,:ndocc]) - np.einsum('ijij->',mo_eri.np[:ndocc,:ndocc,:ndocc,:ndocc])

print("E[SCF]: ", escf + enuc)
print("E[SCF/Psi4]: ", e)

#form fock matrix in MO basis
f =  np.zeros_like(h,dtype=prec)
f += h
f += 2*np.einsum('pqkk->pq',mo_eri.np[:,:,:ndocc,:ndocc])
f -= np.einsum('pkqk->pq',mo_eri.np[:,:ndocc,:,:ndocc])

#form antisymmetrized MO integrals
mo_eri = mo_eri.np
mo_eri = np.array(mo_eri,dtype=prec)

print('Transforming MO -> antisymmetrized SpO')
ta = time()
tb = time()
print('Done in {} s'.format(tb-ta))

if disk_iJaB:
    print('Using disk arrays')
    mmo_iJaB = np.memmap('iJaB.npy',dtype=prec,mode='write',shape=(mo_eri.shape))
    mmo_iJaB[:] = mo_eri.transpose(0,2,1,3)[:]
    del mmo_iJaB
    mmo_iJAb = np.memmap('iJAb.npy',dtype=prec,mode='write',shape=(mo_eri.shape))
    mmo_iJAb[:] = -mo_eri.transpose(0,3,1,2)[:]
    del mmo_iJAb
    mmo_ijab = np.memmap('ijab.npy',dtype=prec,mode='write',shape=(mo_eri.shape))
    mmo_ijab[:] = mo_eri.transpose(0,2,1,3) - mo_eri.transpose(0,3,1,2)
    del mmo_ijab
    iJAb = np.memmap('iJAb.npy',dtype=prec,mode='r',shape=(mo_eri.shape))
    iJaB = np.memmap('iJaB.npy',dtype=prec,mode='r',shape=(mo_eri.shape))
    ijab = np.memmap('ijab.npy',dtype=prec,mode='r',shape=(mo_eri.shape))
else:
    print('Using in core arrays')
    ijab = mo_eri.transpose(0,2,1,3) - mo_eri.transpose(0,3,1,2)
    iJAb = -mo_eri.transpose(0,3,1,2)
    iJaB = mo_eri.transpose(0,2,1,3)
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
if disk_T2:
    print('Using disk T2')
    tiJaB = np.memmap('T2.npy',dtype=prec,mode='w+',shape=(ndocc,ndocc,nbf-ndocc,nbf-ndocc))
    tiJaB[:] = iJaB[:ndocc,:ndocc,ndocc:,ndocc:]/Dijab
else:
    print('Using in core T2')
    tiJaB = iJaB[:ndocc,:ndocc,ndocc:,ndocc:]/Dijab
tiJAb = iJAb[:ndocc,:ndocc,ndocc:,ndocc:]/Dijab

emp2 = 0.0
emp2 = np.einsum('ijab,ijab->',ijab[:ndocc,:ndocc,ndocc:,ndocc:],tijab)/2
emp2 += np.einsum('ijab,ijab->',iJaB[:ndocc,:ndocc,ndocc:,ndocc:],tiJaB)/2
emp2 += np.einsum('ijab,ijab->',iJAb[:ndocc,:ndocc,ndocc:,ndocc:],tiJAb)/2
print(emp2)

def form_Fae(tia,iJaB,tiJaB):
    temp_1 =  tiJaB + np.einsum('ma,nf->mnaf',tia,tia)/2
    temp_2 =  2*iJaB - iJaB.transpose((1,0,2,3))
    Fae    =  np.zeros( (nbf-ndocc,nbf-ndocc))
    Fae   +=  np.einsum('mf,amef->ae'  , tia   , temp_2[v,o,v,v]   )
    Fae   -=  np.einsum('mnaf,mnef->ae', temp_1, temp_2[o,o,v,v]   )
    return Fae

def form_Fmi(tia,iJaB,tiJaB):
    Fmi    =   np.zeros((ndocc,ndocc))
    temp_1 =   np.einsum( 'ie,nf->inef'  , tia   , tia             )/2
    temp_2 =   2*iJaB - iJaB.transpose(1,0,2,3)
    temp_3 =   2*iJaB - iJaB.transpose(0,1,3,2)
    Fmi   +=   np.einsum( 'ne,mnie->mi'  , tia   , temp_2[o,o,o,v] )
    Fmi   +=   np.einsum( 'inef,mnef->mi', tiJaB , temp_3[o,o,v,v] )
    Fmi   +=   np.einsum( 'inef,mnef->mi',  temp_1, temp_3[o,o,v,v])
    return Fmi

def form_Fme(tia,iJaB):
    Fme  =  np.zeros_like(tia)
    temp_1 = 2*iJaB - iJaB.transpose(1,0,2,3)
    Fme +=  np.einsum( 'nf,mnef->me', tia, temp_1[o,o,v,v])
    return Fme

def form_Wmnij(iJaB,tia,tiJaB):
    Wmnij  = np.zeros_like(iJaB[o,o,o,o])
    Wmnij += iJaB[o,o,o,o]
    temp_1 = np.einsum( 'ie,jf->ijef'    , tia   , tia           )
    Wmnij += np.einsum( 'je,mnie->mnij'  , tia   , iJaB[o,o,o,v] )
    Wmnij += np.einsum( 'ie,mnej->mnij'  , tia   , iJaB[o,o,v,o] )
    Wmnij += np.einsum( 'ijef,mnef->mnij', tiJaB , iJaB[o,o,v,v] )/2
    Wmnij += np.einsum( 'ijef,mnef->mnij', temp_1, iJaB[o,o,v,v] )/2
    return Wmnij

def form_Wabef( iJaB,tia,tiJaB):
    if disk_T2:
        Wabef = np.memmap('Wabef.npy',dtype=prec,mode='w+',shape=iJaB[v,v,v,v].shape)
    else:
        Wabef  = np.zeros_like(iJaB[v,v,v,v])
    Wabef += iJaB[v,v,v,v]
    temp_1 = np.einsum( 'ma,nb->mnab'    , tia   , tia           )
    Wabef -= np.einsum( 'mb,amef->abef'  , tia   , iJaB[v,o,v,v] )
    Wabef -= np.einsum( 'ma,mbef->abef'  , tia   , iJaB[o,v,v,v] )
    Wabef += np.einsum( 'mnab,mnef->abef', tiJaB , iJaB[o,o,v,v] )/2
    Wabef += np.einsum( 'mnab,mnef'      , temp_1, iJaB[o,o,v,v] )/2
    return Wabef

def form_WmBeJ(iJaB,tia,tiJaB):
    if disk_T2:
        WmBeJ = np.memmap('WmBeJ.npy',dtype=prec,mode='w+',shape=iJaB[o,v,v,o].shape)
    else:
        WmBeJ  = np.zeros_like(iJaB[o,v,v,o])
    WmBeJ += iJaB[o,v,v,o]
    temp_1 = np.einsum( 'jf,nb->jnfb'    , tia     , tia                                 )
    WmBeJ += np.einsum( 'jf,mbef->mbej'  , tia     , iJaB[o,v,v,v]   )
    WmBeJ -= np.einsum( 'nb,mnej->mbej'  , tia     , iJaB[o,o,v,o]   )
    WmBeJ -= np.einsum( 'jnfb,mnef->mbej', tiJaB   , iJaB[o,o,v,v]   )/2
    WmBeJ -= np.einsum( 'jnfb,mnef->mbej', 2*temp_1, iJaB[o,o,v,v]   )/2
    WmBeJ += np.einsum( 'njfb,mnef->mbej', tiJaB   , 2*iJaB[o,o,v,v] )/2
    WmBeJ -= np.einsum( 'njfb,nmef->mbej', tiJaB   , iJaB[o,o,v,v]   )/2
    return WmBeJ

def form_WmBEj(iJaB,tia,tiJaB):
    if disk_T2:
        WmBEj = np.memmap('WmBEj.npy',dtype=prec,mode='w+',shape=iJaB[o,v,v,o].shape)
    else:
        WmBEj  = np.zeros_like(iJaB[o,v,v,o])
    temp_1 = np.einsum('jf,nb->jnfb',tia,tia)
    WmBEj -= iJaB.transpose((1,0,2,3))[o,v,v,o]
    WmBEj -= np.einsum( 'jf,mbfe->mbej'  , tia   , iJaB[o,v,v,v] )
    WmBEj += np.einsum( 'nb,nmej->mbej'  , tia   , iJaB[o,o,v,o] )
    WmBEj += np.einsum( 'jnfb,nmef->mbej', tiJaB + 2*temp_1, iJaB[o,o,v,v] )/2
    #WmBEj += np.einsum( 'jnfb,nmef->mbej', temp_1, iJaB[o,o,v,v] )
    return WmBEj

def update_T1(tia,Fae,Fme,Fmi,tiJaB,iJaB):
    _tia  = np.zeros_like(tia)
    _tia += np.einsum( 'ie,ae->ia'     ,tia   , Fae             )
    _tia -= np.einsum( 'ma,mi->ia'     ,tia   , Fmi             )
    _tia += np.einsum( 'me,imae->ia'   ,Fme   , 2*tiJaB - tiJaB.transpose(1,0,2,3)        )
    #_tia -= np.einsum( 'me,miae->ia'   ,Fme   , tiJaB           )
    _tia += np.einsum( 'me,amie->ia'   ,tia   , 2*iJaB[v,o,o,v] )
    _tia -= np.einsum( 'me,maie->ia'   ,tia   , iJaB[o,v,o,v]   )
    _tia -= np.einsum( 'mnae,mnie->ia' ,tiJaB , 2*iJaB[o,o,o,v] - iJaB[o,o,o,v].transpose(1,0,2,3))
    #_tia += np.einsum( 'mnae,nmie->ia' ,tiJaB , iJaB[o,o,o,v]   )
    _tia += np.einsum( 'imef,amef->ia' ,tiJaB , 2*iJaB[v,o,v,v] - iJaB[v,o,v,v].transpose(0,1,3,2))
    _tia -= np.einsum( 'imef,amfe->ia' ,tiJaB , iJaB[v,o,v,v]   )
    _tia /= Dia
    return _tia

def update_T2(tia,Fae,Fme,Fmi,tiJaB,iJaB):
    "Equation 47"
    if disk_T2:
        _tiJaB = np.memmap('_T2.npy',dtype=prec,mode='w+',shape=((ndocc,ndocc,nbf-ndocc,nbf-ndocc)))
        _tiJaB[:] = 0
    else:
        _tiJaB = np.zeros_like(tiJaB)
    _tiJaB += iJaB[o,o,v,v]

    #Fae = form_Fae(tia,iJaB,tiJaB)
    #Fmi = form_Fmi(tia,iJaB,tiJaB)
    #Fme = form_Fme(tia,iJaB)
    _tiJaB += np.einsum( 'ijae,be->ijab', tiJaB, Fae  , optimize=True)
    _tiJaB += np.einsum('ijeb,ae->ijab',tiJaB,Fae, optimize=True)
    temp    = np.einsum( 'mb,me->be',     tia,   Fme  , optimize=True)/2
    _tiJaB -= np.einsum( 'ijae,be->ijab', tiJaB, temp , optimize=True)
    temp = np.einsum('ma,me->ae',tia,Fme, optimize=True)/2
    _tiJaB -= np.einsum('ijeb,ae->ijab',tiJaB,temp, optimize=True)

    _tiJaB -= np.einsum('imab,mj->ijab',tiJaB,Fmi, optimize=True)
    _tiJaB -= np.einsum('mjab,mi->ijab',tiJaB,Fmi, optimize=True)

    temp = np.einsum('je,me->mj',tia,Fme, optimize=True)/2
    _tiJaB -= np.einsum('imab,mj->ijab',tiJaB,temp, optimize=True)
    temp = np.einsum('ie,me->mi',tia,Fme, optimize=True)/2
    _tiJaB -= np.einsum('mjab,mi->ijab',tiJaB,temp, optimize=True)
    del Fmi
    del Fme
    del temp

    temp_1  = np.einsum('ma,nb->mnab',tia,tia, optimize=True)
    _tiJaB -= np.einsum('jmea,bmei->ijab',temp_1,iJaB[v,o,v,o], optimize=True)
    _tiJaB -= np.einsum('imeb,amej->ijab',temp_1,iJaB[v,o,v,o], optimize=True)
    _tiJaB -= np.einsum('imea,mbej->ijab',temp_1,iJaB[o,v,v,o], optimize=True)

    Wmnij = form_Wmnij(iJaB,tia,tiJaB)
    _tiJaB += np.einsum('mnab,mnij->ijab',(tiJaB + temp_1),Wmnij, optimize=True)
    del Wmnij

    Wabef = form_Wabef(iJaB,tia,tiJaB)
    _tiJaB += np.einsum('ijef,abef->ijab',(tiJaB + temp_1),Wabef, optimize=True)
    del Wabef

    WmBeJ = form_WmBeJ(iJaB,tia,tiJaB)
    _tiJaB += np.einsum('imae,mbej->ijab',(tiJaB - tiJaB.transpose(1,0,2,3)),WmBeJ, optimize=True)

    _tiJaB += np.einsum('imae,mbej->ijab',tiJaB,WmBeJ , optimize=True) 
    _tiJaB += np.einsum('jmbe,maei->ijab',(tiJaB - tiJaB.transpose(1,0,2,3)),WmBeJ, optimize=True)
    _tiJaB += np.einsum('jmbe,maei->ijab',tiJaB,WmBeJ, optimize=True)
    del WmBeJ
    WmBEj = form_WmBEj(iJaB,tia,tiJaB)
    _tiJaB += np.einsum('imae,mbej->ijab',tiJaB,(WmBEj), optimize=True)
    _tiJaB += np.einsum('mibe,maej->ijab',tiJaB,WmBEj, optimize=True)
    _tiJaB += np.einsum('mjae,mbei->ijab',tiJaB,WmBEj, optimize=True)
    _tiJaB += np.einsum('jmbe,maei->ijab',tiJaB,WmBEj, optimize=True)

    _tiJaB -= np.einsum('jmeb,maei->ijab',temp_1,iJaB[o,v,v,o], optimize=True)
    _tiJaB += np.einsum('ie,abej->ijab',tia,iJaB[v,v,v,o],optimize=True)
    _tiJaB += np.einsum('je,abie->ijab',tia,iJaB[v,v,o,v],optimize=True)
    _tiJaB -= np.einsum('ma,mbij->ijab',tia,iJaB[o,v,o,o],optimize=True)
    _tiJaB -= np.einsum('mb,amij->ijab',tia,iJaB[v,o,o,o],optimize=True)
    _tiJaB /= Dijab
    return _tiJaB

def ccenergy(tia,tiJaB,iJaB,tecc):
    t1 = time()
    ecc = 0
    temp_1 = np.einsum('ia,jb->ijab',tia,tia)
    ecc += np.einsum('ijab,ijab->',iJaB[o,o,v,v],2*tiJaB + 2*temp_1 - tiJaB.transpose(1,0,2,3) - temp_1.transpose(1,0,2,3))
    t2 = time()
    tecc += t2 - t1
    return ecc,tecc

def cciter(tia,tiJaB,iJaB,t1p,t2p,tecc):
    t1 = time()
    Fae       = form_Fae(tia,iJaB,tiJaB)
    Fmi       = form_Fmi(tia,iJaB,tiJaB)
    Fme       = form_Fme(tia,iJaB)
    tia_new   = update_T1(tia,Fae,Fme,Fmi,tiJaB,iJaB)
    t2 = time()
    t1p += t2 - t1
    t1 = time()
    tiJaB_new = update_T2(tia,Fae,Fme,Fmi,tiJaB,iJaB)
    t2 = time()
    t2p += t2 - t1
    return tia_new,tiJaB_new,t1p,t2p

e,tecc = ccenergy(tia,tiJaB,iJaB,tecc)
print(e)
for i in range(10):
    if disk_T2:
        tiJaB = np.memmap('T2.npy',dtype=prec,mode='r',shape=((ndocc,ndocc,nbf-ndocc,nbf-ndocc)))
    tia_new,tiJaB_new,t1p,t2p = cciter(tia,tiJaB,iJaB,t1p,t2p,tecc)
    e,tecc = ccenergy(tia_new,tiJaB_new,iJaB,tecc)
    print(e)
    if disk_T2:
        del tiJaB
        tiJaB = np.memmap('T2.npy',dtype=prec,mode='write',shape=((ndocc,ndocc,nbf-ndocc,nbf-ndocc)))
    tia[:] = tia_new[:]
    tiJaB[:] = tiJaB_new[:]
    if disk_T2:
        tiJaB.flush()
        del tiJaB
        del tia_new
        del tiJaB_new
    #tia,tiJaB = cciter(tia,tiJaB,iJaB)
print('T[1particle] : ', t1p)
print('T[2particle] : ', t2p)
print('T[ccenergy_] : ', tecc)

