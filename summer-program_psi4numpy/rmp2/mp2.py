
import numpy as np
import psi4
import sys
import time

sys.path.append('../../3/mdav2')
#import rhf

def mp2(moeri, e, nocc):
#Calculates mp2 (moller-plesset @ 2nd order) correlation energy.
#Takes in a coefficient matrix (C), two-electron integrals (g),
#orbital energies (e), and an occupation number (nocc)

    moeri_alt = moeri.transpose(0,1,3,2)
    #moeri_alt = moeri
    #moeri_alt = np.zeros_like(moeri)
    #moeri_alt = moeri
    mp2_correction = 0
#Calculate mp2 corr energy via the RMP2 simplifications
    for I in range(nocc):
        for J in range(nocc):
            for A in range(nocc,len(e)):
                for B in range(nocc, len(e)):
                    mp2_correction += (moeri[I][J][A][B]*(2*moeri[I][J][A][B]\
                                   - moeri_alt[I][J][A][B]))\
                                   /(e[I]+e[J]- e[A] - e[B])
    return mp2_correction

def tei(C, g, do_opt=True):
    t1 = time.time()
    moeri = np.einsum('pQRS,pP->PQRS',
          np.einsum('pqRS,qQ->pQRS',
          np.einsum('pqrS,rR->pqRS',
          np.einsum('pqrs,sS->pqrS', g, C, optimize=do_opt), C, optimize=do_opt), C, optimize=do_opt), C, optimize=do_opt)
    t2 = time.time()
    return moeri, t2 - t1

def transform_tei_forloop_n5(gao, C):
  norb = gao.shape[0]
  g1 = np.zeros(gao.shape)
  g2 = np.zeros(gao.shape)
  for i in range(norb):
    print(i)
  for s in range(norb):
    for si in range(norb):
      for rh in range(norb):
        for nu in range(norb):
          for mu in range(norb):
            g1[mu,nu,rh,s] += gao[mu,nu,rh,si] * C[si,s]
  print(g1)
  for mu in range(norb):
    for nu in range(norb):
      for r in range(norb):
        for s in range(norb):
          for rh in range(norb):
            g2[mu,nu,r,s] += g1[mu,nu,rh,s] * C[rh,r]
  g1.fill(0)
  print(g2)
  for mu in range(norb):
    for q in range(norb):
      for r in range(norb):
        for s in range(norb):
          for nu in range(norb):
            g1[mu,q,r,s] += g2[mu,nu,r,s] * C[nu,q]
  g2.fill(0)
  for p in range(norb):
    for q in range(norb):
      for r in range(norb):
        for s in range(norb):
          for mu in range(norb):
            g2[p,q,r,s] += g1[mu,q,r,s] * C[mu,p]
  return g2

if __name__ == '__main__':
    mol = psi4.geometry("""
    H
    H 1 0.96
    symmetry c1
    """)

    #O
    #H 1 0.96
    #H 1 0.96 2 104.5
    bbasis = 'cc-pvtz'
    psi4.set_options({'scf_type':'pk',
                      'basis':bbasis})
    psi4.core.be_quiet()
    EHF,wfn = psi4.energy(f'hf/{bbasis}',mol=mol,return_wfn=True)
    C = np.asarray(wfn.Ca())

    basisname = psi4.core.get_global_option("BASIS")
    basis = psi4.core.BasisSet.build(mol, target=str(basisname), key='basis')
    mints = psi4.core.MintsHelper(basis)
    g = np.asarray(mints.ao_eri())
    print(g)
    #moeri,t = tei(C,g)
    t1 = time.time()
    moeri,dt1 = tei(C,g)#$transform_tei_forloop_n5(g,C)
    moeri,dt2 = tei(C,g,do_opt=False)
    #print(moeri)
    #exit()
    #mints = psi4.core.MintsHelper(wfn)
    #moeri = np.asarray(mints.mo_eri(C,C,C,C)).transpose(0,2,1,3)
    e = wfn.epsilon_a().to_array()
    nocc = wfn.nalpha()
    #print(nocc)
    #print(moeri)
    #print(e)
    #EHF, msg, C, g, e, nocc = rhf.rhf(mol, basis='cc-sto-3g', iterlim=500)
    mp2_corr = mp2(moeri, e, nocc )
    E =EHF + mp2_corr
    t2 = time.time()
    dt = t2 - t1
    print("EHF: {}\nMP2 CORRECTION: {}\nE=(EHF+MP2): {}".format(EHF, mp2_corr, E))
    print('MP2 correction should be: {}'.format(-0.03572552015351))
    print('time: {}'.format(dt))
    print('optimized tei transorm: ', dt1)
    print('non-optimized tei transform: ', dt2)
