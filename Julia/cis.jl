using PyCall
using LinearAlgebra 
using Dates
const psi4 = pyimport("psi4")
const np = pyimport("numpy")
const dt = Float64

function init(mol,basis)
    psi4.core.be_quiet()
    psi4.set_options(Dict("scf_type" => "pk",
    		              "basis" => basis))
    ehf,wfn = psi4.energy("hf",mol=mol,return_wfn=true)
    return wfn
end


@inline @inbounds @fastmath function HS(eri::Array{dt,4}, F::Array{dt,2}, i::Int64, j::Int64, a::Int64, b::Int64)
    return kron(i,j)*F[a,b] - kron(a,b)*F[i,j] + so_eri(eri,a,j,i,b)
end

function setup(wfn)
    _C = wfn.Ca()
    nbf = wfn.nmo()
    nso = 2*nbf
    nocc = 2*wfn.nalpha()
    eps = wfn.epsilon_a().to_array()
    nvir = nso - nocc
    basis = wfn.basisset()
    mints = psi4.core.MintsHelper(basis)
    mo_eri = convert(Array{dt},mints.mo_eri(_C,_C,_C,_C).to_array())
    ff = zeros(dt,nso,nso)
    r = collect(UnitRange(1,nso))
    @inbounds @fastmath for i in r
        ff[i,i] = eps[Int64(fld((i+1),2))]
    end
    return nocc,nvir,mo_eri,ff
end

@inbounds @fastmath function make_H(nocc,nvir,so_eri,F)
    H = zeros(dt,nocc*nvir,nocc*nvir)
    rocc = collect(UnitRange(1,nocc))
    rvir = collect(UnitRange(1,nvir))
    for b in rvir
        for i in rocc
            for j in rocc
                for a in rvir
                    aa = a + nocc
                    bb = b + nocc
                    I = (i-1)*nvir+a
                    J = (j-1)*nvir+b
                    H[I,J] = HS(so_eri,F,i,j,aa,bb)
                end
            end
        end
    end
    return H
end

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

function cis(wfn,nroots)
    t0 = Dates.Time(Dates.now())
    nocc,nvir,so_eri,F = setup(wfn)
    t1 = Dates.Time(Dates.now())
    print("Setup completed in ") 
    print(convert(Dates.Millisecond,(t1 - t0)))
    print("\n")
    t0 = Dates.Time(Dates.now())
    H = make_H(nocc,nvir,so_eri,F)
    H = Symmetric(H)
    t1 = Dates.Time(Dates.now())
    print("Hamiltonian constructed in ") 
    print(convert(Dates.Millisecond,(t1 - t0)))
    print("\n")
    t0 = Dates.Time(Dates.now())
    eigs = eigvals(H,1:nroots)
    t1 = Dates.Time(Dates.now())
    print("Hamiltonian diagonalized in ") 
    print(convert(Dates.Millisecond,(t1 - t0)))
    print("\n")
    return eigs[1:nroots]
end

@views @inline @fastmath @inbounds function so_eri(mo_eri,p,r,q,s)
    pp = Int64(fld((p+1),2))
    qq = Int64(fld((q+1),2))
    rr = Int64(fld((r+1),2))
    ss = Int64(fld((s+1),2))
    return (p%2==q%2)*(r%2==s%2)*mo_eri[pp,qq,rr,ss] - (p%2==s%2)*(q%2==r%2)*mo_eri[pp,ss,qq,rr]
end

@inline @fastmath function kron(a::Int64,b::Int64)
    "Kronecker delta function"
    @fastmath return a == b
end


mol = psi4.geometry("""
                    pubchem:ethane
                    symmetry c1
                    """)
wfn = init(mol,"cc-pvdz")
print(cis(wfn,2))
