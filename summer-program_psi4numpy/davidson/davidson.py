#closely adapted from https://joshuagoings.com/2013/08/23/davidsons-method/
import numpy as np
import time

def Davidson(A,eigs=1,k=4,kmax=20,tol=1E-8):
    """Simple davidson solver
    No restarts, so set kmax sufficiently high.
    If not converged, returns (False, False)
    inputs
        A : matrix to find eigensolutions of
        eigs : (default 1) number of eigenvalues requested
        k : (default 4) trial subspace expansion size
        kmax : (default 20) trial subspace max size
        tol : (default 1E-8) convergence criterion for eigenvalues
    outputs -> (val, vec)
        val : approximate eigenvalues
        vec : corresponding approximate eigenvectors
    Go dawgs"""
    assert A.shape[0] == A.shape[1], print("Davidson received a non-square matrix.")
    n = A.shape[0]
    t = np.eye(n,k) #to draw guess vectors from
    V = np.zeros((n,n)) #storage of trial subspace
    I = np.eye(n) #identity duh
    for m in range(k,kmax,k):
        if m <= k:
            l_old = 1
            for j in range(k):
                V[:,j] = t[:,j]#/np.linalg.norm(t[:,j])
                               # ^ - - - we don't need to do this, there is
                               # a QR decomposition performed on V
                               #below     +
        else:                  #          '
            l_old = l[:eigs]   #          '
                               #          '
        V,R = np.linalg.qr(V)  #< - - - - +
        T = np.dot(V[:,:(m+1)].T,np.dot(A,V[:,:(m+1)]))
        L,E = np.linalg.eig(T)
        idx = L.argsort()
        l = L[idx]
        e = E[:,idx]
        
        for j in range(k):
            w = np.dot(A - l[j]*I,np.dot(V[:,:(m+1)],e[j]))
            q = w/(l[j] - A[j,j])
            V[:,j+m+1] = q
        norm = np.linalg.norm(l[:eigs] - l_old)
        if norm < tol:
            return l[:eigs],e[:,:eigs]
    return False,False

if __name__ == "__main__":
    n = 1000
    sparsity = 0.000001
    A = np.zeros((n,n))
    for i in range(n):
        A[i,i] = i + 1
    A = A + sparsity*np.random.randn(n,n)
    A = (A.T + A)/2
    val,vec = Davidson(A)
    print("<<--Eigenvalue comparison-->>")
    if val:
        print("Davidson produces ",val[0])
    print("np.linalg.eigh produces ",np.linalg.eigh(A)[0][0])
    print("-"*20)
    print("delta should be less than 1E-8")
    print("delta: ", val[0] - np.linalg.eigh(A)[0][0])
    print("<<--Eigenvector comparison-->>")
    if val:
        print("Davidson produces ", vec.reshape(len(vec),))
    print("np.linalg.eigh produces ", np.linalg.eigh(A)[1][0][:len(vec)].T)
    print("-"*20)
    print("delta: ", vec.reshape(len(vec),) - np.linalg.eigh(A)[1][0][:len(vec)].T)