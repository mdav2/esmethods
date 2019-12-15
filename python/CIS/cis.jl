using PyCall
using BenchmarkTools
using LinearAlgebra 
using Statistics
psi4 = pyimport("psi4")
np = pyimport("numpy")

function init(bbasis)
    mol = psi4.geometry("""
        O
        H 1 1.1
        H 1 1.1 2 104.0
        symmetry c1""")
    psi4.core.be_quiet()
    basis = bbasis
    psi4.set_options(Dict("scf_type" => "pk",
    		              "basis" => basis))
    ehf,wfn = psi4.energy("hf",mol=mol,return_wfn=true)
    return wfn
end

function SO_eri(mo_eri,nbf)
    #prqs = 
    # d(pq)*d(rs)*pqrs
    #- d(ps)*d(qr)*psqr
    nso = 2*nbf
    so_eri = zeros(Float64,nso,nso,nso,nso)
    range = collect(UnitRange(1,nso))
    @inbounds for s in range
        @inbounds for r in range
            @inbounds for q in range
                @inbounds for p in range
                    @fastmath pp = Int64(floor((p+1)/2))
                    @fastmath qq = Int64(floor((q+1)/2))
                    @fastmath rr = Int64(floor((r+1)/2))
                    @fastmath ss = Int64(floor((s+1)/2))
                    @inbounds @fastmath spint1 = (p%2==q%2)*(r%2==s%2)*mo_eri[pp,qq,rr,ss]
                    @inbounds @fastmath so_eri[p,r,q,s] += spint1
                end
            end
        end
    end
    @inbounds for r in range
        @inbounds for q in range
            @inbounds for s in range
                @inbounds for p in range
                    @fastmath pp = Int64(floor((p+1)/2))
                    @fastmath qq = Int64(floor((q+1)/2))
                    @fastmath rr = Int64(floor((r+1)/2))
                    @fastmath ss = Int64(floor((s+1)/2))
                    @inbounds @fastmath spint2 = (p%2==s%2)*(q%2==r%2)*mo_eri[pp,ss,qq,rr]
                    @inbounds @fastmath so_eri[p,r,q,s] -= spint2
                end
            end
        end
    end
    return so_eri
end

function kron(a::Int64,b::Int64)
    @fastmath return a == b
end

function HS(eri::Array{Float64,4}, F::Array{Float64,2}, i::Int64, j::Int64, a::Int64, b::Int64)
    @inbounds @fastmath return kron(i,j)*F[a,b] - kron(a,b)*F[i,j] + eri[a,j,i,b]
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
    mo_eri = mints.mo_eri(_C,_C,_C,_C).to_array()::Array{Float64}
    so_eri = SO_eri(mo_eri,nbf)::Array{Float64}
    ff = zeros(Float64,nso,nso)
    r = collect(UnitRange(1,nso))
    @inbounds for i in r
        @fastmath ff[i,i] = eps[Int64(floor((i+1)/2))]
    end
    return nocc,nvir,so_eri,ff
end

function make_H(nocc,nvir,so_eri,F)
    H = zeros(Float64,nocc*nvir,nocc*nvir)
    rocc = collect(UnitRange(1,nocc))
    rvir = collect(UnitRange(1,nvir))
    @inbounds for b in rvir
        @inbounds for i in rocc
            @inbounds for j in rocc
                @inbounds for a in rvir
                    @fastmath aa = a + nocc
                    @fastmath bb = b + nocc
                    @fastmath I = (i-1)*nvir+a
                    @fastmath J = (j-1)*nvir+b
                    @fastmath H[I,J] = HS(so_eri,F,i,j,aa,bb)
                end
            end
        end
    end
    return H
end

function cis(wfn)
    nocc,nvir,so_eri,F = setup(wfn)
    H = make_H(nocc,nvir,so_eri,F)
    eigs = eigvals(H)
    return eigs[1]
end
#wfn = init()
#println(cis(wfn))
#t = @benchmark cis(wfn)
#println(t)
