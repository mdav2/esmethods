#very naive direct MP2
import numpy as np
import psi4
psi4.core.be_quiet()

mol = psi4.geometry("""
        O
        H 1 1.1
        H 1 1.1 2 104.0
        symmetry c1
        """)

psi4.set_options({'basis':'sto-3g',
                  'scf_type':'pk'})

e,wfn = psi4.energy('scf',mol=mol,return_wfn=True)
_C = wfn.Ca()
nbf = wfn.nmo()
nmo = nbf
nocc = wfn.nalpha()
eps = wfn.epsilon_a().to_array()
C = _C.to_array()
mints = psi4.core.MintsHelper(wfn.basisset())

basis = wfn.basisset()
#print(basis.nshell_on_center(0))

integ = mints.integral()
si = integ.shells_iterator()
si.first()
ao_eri = np.zeros((7,7,7,7))
while not si.is_done():
    p,q,r,s = (si.p,si.q,si.r,si.s)
    pf = basis.shell_to_basis_function(p)
    qf = basis.shell_to_basis_function(q)
    rf = basis.shell_to_basis_function(r)
    sf = basis.shell_to_basis_function(s)
    shell = mints.ao_eri_shell(p,q,r,s).to_array()
    pn,qn,rn,sn = shell.shape
    ao_eri[pf:pf+pn,qf:qf+qn,rf:rf+rn,sf:sf+sn] = shell

    ao_eri[qf:qf+qn,pf:pf+pn,rf:rf+rn,sf:sf+sn] = shell.transpose(1,0,2,3)

    ao_eri[pf:pf+pn,qf:qf+qn,sf:sf+sn,rf:rf+rn] = shell.transpose(0,1,3,2)

    ao_eri[qf:qf+qn,pf:pf+pn,sf:sf+sn,rf:rf+rn] = shell.transpose(1,0,2,3).transpose(0,1,3,2)

    ao_eri[rf:rf+rn,sf:sf+sn,pf:pf+pn,qf:qf+qn] = shell.transpose(2,1,0,3).transpose(0,3,2,1)

    ao_eri[sf:sf+sn,rf:rf+rn,pf:pf+pn,qf:qf+qn] = shell.transpose(2,1,0,3).transpose(0,3,2,1).transpose(1,0,2,3)

    ao_eri[rf:rf+rn,sf:sf+sn,qf:qf+qn,pf:pf+pn] = shell.transpose(2,1,0,3).transpose(0,3,2,1).transpose(0,1,3,2)

    ao_eri[sf:sf+sn,rf:rf+rn,qf:qf+qn,pf:pf+pn] = shell.transpose(3,1,2,0).transpose(0,2,1,3)

    si.next()

def ao(mints,basis,p,q,r,s):
    ps = basis.function_to_shell(p)
    qs = basis.function_to_shell(q)
    rs = basis.function_to_shell(r)
    ss = basis.function_to_shell(s)
    
    po = p - basis.shell_to_ao_function(ps)
    qo = q - basis.shell_to_ao_function(qs)
    ro = r - basis.shell_to_ao_function(rs)
    so = s - basis.shell_to_ao_function(ss) 

    shell = mints.ao_eri_shell(ps,qs,rs,ss).to_array()
    return shell[po,qo,ro,so]

def mo(mints,basis,C,i,j,k,l):
    summ = 0

    #for p in range(nbf):
    #    for q in range(nbf):
    #        for r in range(nbf):
    #            for s in range(nbf):
    #                summ += C[p,i]*C[q,j]*ao(mints,basis,p,q,r,s)*C[r,k]*C[s,l]
    return summ

emp2 = 0
for i in range(nocc):
    for j in range(nocc):
        for a in range(nocc,nmo):
            for b in range(nocc,nmo):
                temp = mo(mints,basis,C,i,a,j,b)
                emp2 += (temp*(2*temp - mo(mints,basis,C,i,b,j,a)))/(eps[i] + eps[j] - eps[a] - eps[b])
print(emp2)
