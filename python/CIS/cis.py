import numpy as np
import os
import psi4
import time

def kron( a , b ):
    "kronecker delta"
    return a == b

def HS(eri, F, i, j, a, b):
    "computes a singly substituted hamiltonian matrix element"
    return kron(i,j)*F[a][b] - kron(a,b)*F[i][j] + eri[a][j][i][b]

def wfn_info(mol,wftype='scf',basis='sto-3g',symmetry=False):
    "takes in a molecule, returns wfn info needed for CIS"
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

    return({'f':f,'mints':mints,'h':h,'eps':eps,'C':C,'mo_eri':mo_eri,'o':o,'v':v,'nocc':nocc,'nmo':nmo,'nvir':nvir})

def make_so_eri(info,disk=True,algo="fast"):
    print("Allocating spin orbital integrals...")
    nmo = info['nmo']
    mo_eri = info['mo_eri']
    if disk:
        so_eri = np.memmap(filename="soeri.npz",mode="w+",dtype=dt,shape=(nmo,nmo,nmo,nmo))
        so_eri[:] = 0
    else:
        so_eri = np.zeros((nmo,nmo,nmo,nmo))
    print("Transforming integrals ...")
    if algo == "fast":
        temp1 = mo_eri.transpose(0,2,1,3)
        temp2 = mo_eri.transpose(0,3,2,1)
        so_eri[0::2,0::2,0::2,0::2] += temp1 - temp2
        print("16.6%")
        # p = q = 0; r = s = 1
        so_eri[0::2,1::2,0::2,1::2] += temp1 
        print("33.3%")
        # p = q = 1; r = s = 0
        so_eri[1::2,0::2,1::2,0::2] += temp1 
        print("50.0%")
        # p = q = 1; r = s = 1
        so_eri[1::2,1::2,1::2,1::2] += temp1 - temp2
        print("66.6%")

        # p = r & q = s
        # p = r = 1; q = s = 0
        so_eri[1::2,1::2,0::2,0::2] -= temp2
        print("83.3%")
        # p = r = 0; q = s = 1 
        so_eri[0::2,0::2,1::2,1::2] -= temp2
        print("100%")
    else:
        temp1 = mo_eri.transpose(0,2,1,3)
        so_eri[0::2,0::2,0::2,0::2] += temp1
        print("12.5%")
        # p = q = 0; r = s = 1
        so_eri[0::2,1::2,0::2,1::2] += temp1 
        print("25.0%")
        # p = q = 1; r = s = 0
        so_eri[1::2,0::2,1::2,0::2] += temp1 
        print("37.5%")
        # p = q = 1; r = s = 1
        so_eri[1::2,1::2,1::2,1::2] += temp1 
        print("50.0%")

        del temp1
        temp2 = mo_eri.transpose(0,3,2,1)
        so_eri[0::2,0::2,0::2,0::2] -=  temp2 
        print("62.5%")
        so_eri[1::2,1::2,1::2,1::2] -=  temp2 
        print("75.0%")
        so_eri[1::2,1::2,0::2,0::2] -= temp2
        print("87.5%")
        # p = r = 0; q = s = 1 
        so_eri[0::2,0::2,1::2,1::2] -= temp2
        print("100.%")
    if disk:
        so_eri.flush()
        del so_eri
        return 0
    else:
        return so_eri

def get_core(mol):
    core  = 0
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
    return core

def do_cis(mol,nroot=5,dt="float32",freeze_core=True,disk=False,algo="fast"):
    info = wfn_info(mol) 
    if freeze_core:
        core = get_core(mol)
    t1 = time.time()
    so_eri = make_so_eri(info,disk=disk,algo=algo)
    nso = info['nmo']
    nocc = info['nocc']
    nvir = info['nvir']
    eps = info['eps']
    if disk:
        so_eri = np.memmap(filename="soeri.npz",mode="r",dtype=dt,shape=(nso,nso,nso,nso))
    print("integrals transformed, cleaning up ...")
    if disk:
        so_eri.flush()
        del so_eri
    t2 = time.time()
    print("integrals transformed in {} s".format(t2 - t1))


    t1 = time.time()
    H = np.zeros(((nocc - core)*nvir, (nocc - core)*nvir))
    ff = np.zeros((nso,nso))
    for i in range(nso):
        ff[i][i] = eps[int(i/2)]

    if disk:
        for i in range(nocc - core):
            for j in range(nocc - core):
                for a in range(nvir):
                    so_eri = np.memmap(filename="soeri.npz",mode="r",dtype=dt,shape=(nso,nso,nso,nso))
                    aa = a + nocc
                    bb = np.arange(nvir) + nocc
                    H[i*nvir + a][j*nvir:j*nvir+nvir] = HS(so_eri,ff,i+core,j+core,aa,bb)
                    del so_eri
    else:
        for i in range(nocc - core):
            for j in range(nocc - core):
                for a in range(nvir):
                    aa = a + nocc
                    bb = np.arange(nvir) + nocc
                    H[i*nvir + a][j*nvir:j*nvir+nvir] = HS(so_eri,ff,i+core,j+core,aa,bb)
    del ff

    t2 = time.time()
    print('matrix generated in {} s'.format(t2 - t1))
    t1 = time.time()
    print(np.linalg.eigh(H)[0][:nroot])
    t2 = time.time()
    print('matrix diagonalized in {}s'.format(t2 - t1))

if __name__ == "__main__":
    psi4.core.be_quiet()
    mol = psi4.geometry("""
    O
    H 1 1.1
    H 1 1.1 2 104.0
    symmetry c1
    """)
    nat = mol.natom()
    psi4.set_options({'basis':'cc-pvtz',
                      'e_convergence':10,
                      'd_convergence':10,
                      'scf_type':'pk'})
    do_cis(mol)
