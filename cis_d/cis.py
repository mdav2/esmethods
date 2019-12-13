import numpy as np
import psi4
import time
from multiprocessing import Pool

def hswrap(intup):
    eri,F,i,j,a,b = intup
    return HS(eri,F,i,j,a,b)

def kron( a , b ):
    return a == b

def HS(eri, F, i, j, a, b):
    return kron(i,j)*F[a][b] - kron(a,b)*F[i][j] + eri[a][j][i][b]

def wfn_info(mol,wftype='scf',basis='sto-3g',symmetry=False):
    e,wfn = psi4.energy(wftype,mol=mol,return_wfn=True)

    nocc = wfn.nalpha()*2
    nmo = wfn.nmo()*2
    nvir = nmo - nocc
    o = slice(0,nocc)
    v = slice(nocc,nmo)

    mints = psi4.core.MintsHelper(wfn.basisset())
    _C = wfn.Ca()
    C = _C.to_array()
    mo_eri = mints.mo_eri(_C,_C,_C,_C).np
    h = wfn.H().to_array()
    eps = wfn.epsilon_a().np

    f =  np.zeros_like(h)
    f += h
    f += 2*np.einsum('pqkk->pq',mo_eri[:,:,o,o])
    f -= np.einsum('pkqk->pq',mo_eri[:,o,:,o])

    return({'f':f,'h':h,'eps':eps,'C':C,'mo_eri':mo_eri,'o':o,'v':v,'nocc':nocc,'nmo':nmo,'nvir':nvir})

def CIS(mol,basis='sto-3g'):
    info = wfn_info(mol,basis=basis)
    f = info['f']
    h = info['h']
    nocc = info['nocc']
    nvir = info['nvir']
    nmo = info['nmo']

def make_SO_eri(info):
    nmo = info['nmo']
    so_eri = np.zeros((nmo,nmo,nmo,nmo))

if __name__ == "__main__":
    t1 = time.time()
    psi4.core.be_quiet()
    mol = psi4.geometry("""
    pubchem:propane
    symmetry c1
    """)
    nat = mol.natom()
    core = 0
    for i in range(nat):
        if mol.Z(i) in range(2,11):
            core += 1
        elif mol.Z(i) in range(11,20):
            core += 5
        elif mol.Z(i) in range(0,3):
            core += 0
        else:
            print("WTF are you doing? use a real CIS code!")
            exit()
    print(core)
    nroot = 5
    psi4.set_options({'basis':'cc-pvdz',
                      'e_convergence':10,
                      'd_convergence':10,
                      'scf_type':'pk'})
    e,wfn = psi4.energy('hf',mol=mol,return_wfn=True)
    eps = wfn.epsilon_a().np
    nocc = wfn.nalpha()*2
    nmo = wfn.nmo()*2
    nvir = nmo - nocc
    mints = psi4.core.MintsHelper(wfn.basisset())
    _C = wfn.Ca()
    C = _C.to_array()
    mo_eri = mints.mo_eri(_C,_C,_C,_C).np
    o = slice(0,nocc)
    v = slice(nocc,nmo)
    h = wfn.H().to_array()

    f =  np.zeros_like(h)
    f += h
    f += 2*np.einsum('pqkk->pq',mo_eri[:,:,o,o])
    f -= np.einsum('pkqk->pq',mo_eri[:,o,:,o])
    t2 = time.time()
    print('setup completed in {} s'.format(t2 - t1))
    #info = wfn_info(mol)

    #mo_eri = mo_eri.transpose(0,2,1,3)
    #mo_eri = mo_eri - mo_eri.transpose(0,1,3,2)

    t1 = time.time()
    nso = nmo
    so_eri = np.zeros((nso,nso,nso,nso))

    #        p   r   q   s
    # p = q & r = s
    # p = q = 0; r = s = 0
    so_eri[0::2,0::2,0::2,0::2] += mo_eri.transpose(0,2,1,3)  
    # p = q = 0; r = s = 1
    so_eri[0::2,1::2,0::2,1::2] += mo_eri.transpose(0,2,1,3) 
    # p = q = 1; r = s = 0
    so_eri[1::2,0::2,1::2,0::2] += mo_eri.transpose(0,2,1,3) 
    # p = q = 1; r = s = 1
    so_eri[1::2,1::2,1::2,1::2] += mo_eri.transpose(0,2,1,3) 

    # p = r & q = s
    # p = r = 0; q = s = 0
    so_eri[0::2,0::2,0::2,0::2] -= mo_eri.transpose(0,3,2,1)
    # p = r = 1; q = s = 1 
    so_eri[1::2,1::2,1::2,1::2] -= mo_eri.transpose(0,3,2,1)
    # p = r = 1; q = s = 0
    so_eri[1::2,1::2,0::2,0::2] -= mo_eri.transpose(0,3,2,1)
    # p = r = 0; q = s = 1 
    so_eri[0::2,0::2,1::2,1::2] -= mo_eri.transpose(0,3,2,1)

    t2 = time.time()
    print("integrals transformed in {} s".format(t2 - t1))


    t1 = time.time()
    H = np.zeros(((nocc - core)*nvir, (nocc - core)*nvir))
    ff = np.zeros((nso,nso))
    for i in range(nso):
        ff[i][i] = eps[int(i/2)]

    for i in range(nocc - core):
        for j in range(nocc - core):
            for a in range(nvir):
                aa = a + nocc
                bb = np.arange(nvir) + nocc
                H[i*nvir + a][j*nvir:j*nvir+nvir] = HS(so_eri,ff,i+core,j+core,aa,bb)

    t2 = time.time()
    print('matrix generated in {} s'.format(t2 - t1))
    t1 = time.time()
    print(np.linalg.eigh(H)[0][:nroot])
    t2 = time.time()
    print('matrix diagonalized in {}s'.format(t2 - t1))
