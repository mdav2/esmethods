import numpy as np
sparsity = 0.000001  
tol = 1E-7
n = 10
A = np.zeros((n,n))  
for i in range(0,n):  
        A[i,i] = (i + 1)**2 
        #A = A + sparsity*np.random.randn(n,n)  
A = (A.T + A)/2  

k = 4 # number of initial guess vectors  
eig = 1 # number of eignvalues to solve  
t = np.eye(n,k) # set of k unit vectors as guess  
V = np.zeros((n,n)) # array of zeros to hold guess vec  
I = np.eye(n) # identity matrix same dimen as A  
mmax = 40
for m in range(k,mmax,k):
    if m <= k:
        for j in range(0,k):
            V[:,j] = t[:,j]/np.linalg.norm(t[:,j])
        theta_old = 1 
    elif m > k:
        theta_old = theta[:eig]
    V,R = np.linalg.qr(V)
    T = np.dot(V[:,:(m+1)].T,np.dot(A,V[:,:(m+1)]))
    THETA,S = np.linalg.eig(T)
    idx = THETA.argsort()
    theta = THETA[idx]
    s = S[:,idx]
    for j in range(0,k):
        w = np.dot((A - theta[j]*I),np.dot(V[:,:(m+1)],s[:,j])) 
        q = w/(theta[j]-A[j,j])
        V[:,(m+j+1)] = q
    print(V)
    exit()
    #norm = np.linalg.norm(theta[:eig] - theta_old)
    #if norm < tol:
    #    break
