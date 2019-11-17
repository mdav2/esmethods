#!/usr/bin/env python
# coding: utf-8

# In[1]: 

import psi4
import numpy as np
from scipy.sparse import kron
from scipy.linalg import fractional_matrix_power
psi4.set_memory('58 GB')
numpy_memory = 58

if __name__ == "__main__":
    mol = psi4.geometry("""
    H 
    H 1 0.96
    symmetry c1""")
psi4.set_options({'basis':'sto-3g',
                  'scf_type':'pk'})


e_,wfn_ = psi4.energy('scf/6-31G**',return_wfn=True)
epsa = np.asarray(wfn_.epsilon_a())
epsb = np.asarray(wfn_.epsilon_a())
wfn = psi4.core.Wavefunction.build(mol,psi4.core.get_global_option('BASIS'))
ERI_Size = ((2*wfn.nmo())** 4) * 8e-9
print('Size of the ERI/MO tensor will be %4.2f GB.' % ERI_Size)
memory_footprint = ERI_Size * 2.5
if memory_footprint > numpy_memory:
    clean()
    raise Exception("Estimated memory utilization (%4.2f GB) exceeds numpy_memory \
                    limit of %4.2f GB." % (memory_footprint, numpy_memory))
mints = psi4.core.MintsHelper(wfn_.basisset())
Sb = np.asarray(mints.ao_overlap())
Tb = np.asarray(mints.ao_kinetic())
Vb = np.asarray(mints.ao_potential())
Gb0 = np.asarray(mints.ao_eri())
Gb = Gb0.swapaxes(1,2)
nbf = mints.basisset().nbf()
nsbf = 2*nbf
Coa = wfn_.Ca_subset("AO","OCC")
Cob = wfn_.Cb_subset("AO","OCC")
Cva = wfn_.Ca_subset("AO","VIR")
Cvb = wfn_.Ca_subset("AO","VIR")

nocc = int( sum(mol.Z(A) for A in range(mol.natom())) - mol.molecular_charge() )


# In[2]:


Vnu = mol.nuclear_repulsion_energy()


# In[3]:


S = np.block([[Sb,np.zeros(Sb.shape)],
         [np.zeros(Sb.shape),Sb]])
T = np.block([[Tb,np.zeros(Tb.shape)],
         [np.zeros(Tb.shape),Tb]])
V = np.block([[Vb,np.zeros(Vb.shape)],
         [np.zeros(Vb.shape),Vb]])


# In[5]:


def block_tei(A):
    I = np.identity(2)
    A = np.kron(I, A)
    A = A.T
    A = np.kron(I,A)
    return A


def delta(x, y):

    if x < nbf and y < nbf:
        return 1
    elif not x < nbf and not y < nbf:
        return 1
    else:
        return 0
# In[6]:
def block_tei2(A):
    r = range(int(nsbf))
    temp = np.zeros((nsbf, nsbf, nsbf, nsbf))
    for u in r:
        u_i = u % nbf
        for v in r:
            v_i = v % nbf
            for p in r:
                p_i = p % nbf
                for q in r:
                    q_i = q % nbf
                    temp[u, p, v, q] += (delta(u,v) * delta(p,q) * A[u_i, p_i, v_i, q_i] - delta(u,q) * delta(p,v) * A[u_i, p_i, q_i, v_i])

    return temp


#I = np.identity(2)
G = block_tei(Gb)

X = fractional_matrix_power(S,-0.5)

D = np.zeros(S.shape)

H = T + V


G = G.transpose((0,2,1,3)) - G.transpose(0,2,3,1)


# In[7]:



#for i in range(50):
#    v = np.einsum('mrns,sr->mn',G,D)
#
#    F = H + v
#    tf = X.dot(F).dot(X) #X.dot(F.dot(X))
#    e,Ct = np.linalg.eigh(tf)
#    C = X.dot(Ct)#X.dot(Ct)
#    Cocc = C[:, :nocc]
#    D = np.matmul(Cocc, Cocc.T)#Cocc.dot(Cocc.T)
#    Q = H + 0.5*v
#    Ee = np.einsum('uv,vu->',Q,D)
#    E = Ee + Vnu
#    print(E)
#

# In[8]:


def transform_tei(gao, C):
  # g_pqrs = sum_P C_Pp (sum_Q C_Qq (sum_R C_Rr (sum_S C_Ss gao_PQRS)))
  return np.einsum('Pp,Pqrs->pqrs', C,
           np.einsum('Qq,PQrs->Pqrs', C,
             np.einsum('Rr,PQRs->PQrs', C,
               np.einsum('Ss,PQRS->PQRs', C, gao))))

def transform_tei_forloop_n5(gao, C):
  norb = gao.shape[0]
  g1 = np.zeros(gao.shape)
  g2 = np.zeros(gao.shape)
  for mu in range(norb):
    for nu in range(norb):
      for rh in range(norb):
        for s in range(norb):
          for si in range(norb):
            g1[mu,nu,rh,s] += gao[mu,nu,rh,si] * C[si,s]
  for mu in range(norb):
    for nu in range(norb):
      for r in range(norb):
        for s in range(norb):
          for rh in range(norb):
            g2[mu,nu,r,s] += g1[mu,nu,rh,s] * C[rh,r]
  g1.fill(0)
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

# In[9]:


#mo_tei = np.asarray(mints_.mo_eri(Coa,Cva,Cob,Cvb))
mo_tei = transform_tei(G,C)

print(mo_tei)
exit()

# In[10]:


d_mp2 = 0
for I in range(nocc):
    for J in range(nocc):
        for A in range(nocc,len(e)):
            for B in range(nocc,len(e)):
                d_mp2 += (mo_tei[I][J][A][B]**2)/(e[I] + e[J] - e[A] - e[B])
d_mp2 *= 0.25


# In[11]:


print(d_mp2)


# In[ ]:




