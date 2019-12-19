module Davidson
@views function eigdav(A,eigs,k,kmax,tol)
    n = size(A,1)
    V = zeros((n,n))
    t = I(n)
    theta = 0
    w = 0
    theta_old = 0
    for m in k:k:kmax
        if m <= k
            for j in 1:1:k
                V[:,j] = t[:,j]/norm(t[:,j])
            end
            theta_old = ones(eigs)
        else
            theta_old = theta[1:eigs]
        end
        F = qr(V)
        V = Matrix(F.Q)
        T = transpose(V[:,1:(m+1)])*A*V[:,1:(m+1)]
        THETA = eigvals(T)
        S = eigvecs(T)
        idx = sortperm(THETA)
        theta = THETA[idx]
        s = S[:,idx]
        for j in 1:1:k
            w = (A - theta[j]*I(n))*V[:,1:(m+1)]*s[:,j]
            q = w/(theta[j] - A[j,j])
            V[:,m+j+1] = q
        normm = norm(theta[1:eigs] - theta_old)
        if normm < tol
            return theta[1:eigs]
        end
        end

    end
    return theta[1:eigs]
end
end #module Davidson
