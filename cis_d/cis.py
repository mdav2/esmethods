import numpy as np
import psi4
import time

def kron( a , b ):
    assert type(a) == int
    assert type(b) == int
    return a == b

def HS(eri, F, i, j, a, b):
    assert type(F) == np.ndarray
    assert type(eri) == np.ndarray
    assert type(i) == int
    assert type(j) == int
    assert type(a) == int
    assert type(b) == int

    return kron(i,j)*F[a][b] - kron(a,b)*F[i][j] + eri[a][j][i][b]

if __name__ == "__main__":
    psi4.core.be_quiet()
    mol = psi4.geometry("""
    O
    H 1 1.1
    H 1 1.1 2 104.0
    symmetry c1
    """)
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

    #mo_eri = mo_eri.transpose(0,2,1,3)
    #mo_eri = mo_eri - mo_eri.transpose(0,1,3,2)
    nso = nmo
    so_eri = np.zeros((nso,nso,nso,nso))

    t1 = time.time()
    for p in range(nso):
        for q in range(nso):
            for r in range(nso):
                for s in range(nso):
                    pp = int(p/2)
                    qq = int(q/2)
                    rr = int(r/2)
                    ss = int(s/2)

                    spint1 = (p%2 == q%2)*(r%2 == s%2)*mo_eri[pp][qq][rr][ss]
                    spint2 = (p%2 == r%2)*(q%2 == s%2)*mo_eri[pp][ss][qq][rr]
                    so_eri[p][r][q][s] = spint1 - spint2
    t2 = time.time()
    print("integrals transformed in {} s".format(t2 - t1))

    ff = np.zeros((nso,nso))
    for i in range(nso):
        ff[i][i] = eps[int(i/2)]

    H = np.zeros((nocc*nvir, nocc*nvir))
    t1 = time.time()
    for I in range(nocc*nvir):
        for J in range(I,nocc*nvir):
            a = I%nvir + nocc
            i = int((I)/nvir)
            b = J%nvir + nocc
            j = int((J - b)/nvir)
            H[I][J] = HS(so_eri,ff,i,j,a,b)
            H[J][I] = H[I][J]
    #for i in range(nocc):
    #    for a in range(nvir):
    #        for j in range(nocc):
    #            for b in range(nvir):
    #                aa = a + nocc
    #                bb = b + nocc
    #                H[i*nvir + a][j*nvir + b] = HS(so_eri,ff,i,j,aa,bb)
    #                #H[j*nvir + b][i*nvir + a] = HS(so_eri,ff,i,j,aa,bb)
    t2 = time.time()
    print('matrix generated in {} s'.format(t2 - t1))
    print(np.linalg.eigh(H)[0])
