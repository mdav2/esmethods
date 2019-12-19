using PyCall
using BenchmarkTools
psi4 = pyimport("psi4")
np = pyimport("numpy")
mol = psi4.geometry("""
    O
    H 1 0.96
    H 1 0.96 2 104.5
    symmetry c1""")
psi4.core.be_quiet()

bbasis = "cc-pvtz"
psi4.set_options(Dict("scf_type" => "pk",
                      "basis" => bbasis))
ehf,wfn = psi4.energy("hf",mol=mol,return_wfn=true)
basisname = psi4.core.get_global_option("BASIS")
basis = psi4.core.BasisSet.build(mol, target=string(basisname), key="basis")
mints = psi4.core.MintsHelper(basis)
const G = @pycall(np.asarray(mints.ao_eri())::Array{Float64,4})
const C = @pycall(np.asarray(wfn.Ca())::Array{Float64,2})
#time moeri = np.asarray(mints.mo_eri(C,C,C,C))::Array{Float64}
Cp = wfn.Ca()
@benchmark moeri1 = permutedims(np.asarray(mints.mo_eri(Cp,Cp,Cp,Cp))::Array{Float64}, [1,3,2,4])
nocc = wfn.nalpha()::Int64
e = wfn.epsilon_a().to_array()::Array{Float64,1}

function transform_tei2(gao::Array{Float64,4},C::Array{Float64,2})
    norb = size(gao)[1]::Int64 #indexed from 1
    g1 = zeros(size(G))::Array{Float64,4}
    g2 = zeros(size(G))::Array{Float64,4}
    R = collect(UnitRange(1,norb))::Array{Int64,1}
    @inbounds @fastmath for s in R
        @inbounds @fastmath for si in R
            @inbounds @fastmath for rh in R
                @inbounds @fastmath for nu in R
                    @inbounds @fastmath @simd for mu in R
                        @views g1[mu,nu,rh,s] += gao[mu,nu,rh,si]*C[si,s]
                    end
                end
            end
        end
    end
    @inbounds @fastmath for s in R
        @inbounds @fastmath for r in R
            @inbounds @fastmath for rh in R
                @inbounds @fastmath for nu in R
                    @inbounds @fastmath @simd for mu in R
                        @views g2[mu,nu,r,s] += g1[mu,nu,rh,s]*C[rh,r]
                    end
                end
            end
        end
    end
    g1 .= 0.0
    @inbounds @fastmath for s in R
        @inbounds @fastmath for r in R
            @inbounds @fastmath for q in R
                @inbounds @fastmath for nu in R
                    @inbounds @fastmath @simd for mu in R
                        @views g1[mu,q,r,s] += g2[mu,nu,r,s]*C[nu,q]
                    end
                end
            end
        end
    end
    g2 .= 0.0
    @inbounds @fastmath for s in R
        @inbounds @fastmath for r in R
            @inbounds @fastmath for q in R
                @inbounds @fastmath for mu in R
                    @inbounds @fastmath @simd for p in R
                        @views g2[p,q,r,s] += g1[mu,q,r,s]*C[mu,p]
                    end
                end
            end
        end
    end
    return g2
end
function mp2(moeri::Array{Float64,4},e::Array{Float64,1},nocc::Int64)
    moeri_alt = permutedims(moeri,[1,2,4,3])::Array{Float64,4}
    delta_mp2 = 0.0::Float64
    for I in 1:1:nocc
        for J in 1:1:nocc
            for A in nocc+1:1:length(e)
                for B in nocc+1:1:length(e)
                    delta_mp2 += (moeri[I,J,A,B]*(2*moeri[I,J,A,B] - moeri_alt[I,J,A,B]))/(e[I] + e[J] - e[A] - e[B])
                end
            end
        end
    end
    return delta_mp2
end
println("Transforming TEI")
@benchmark moeri = transform_tei2(permutedims(G::Array{Float64,4},[1,3,2,4]), C::Array{Float64,2})
#moeri2 = transform_tei2(G::Array{Float64,4},C::Array{Float64,2})
println("Computing MP2 Correction")
mp2_corr = mp2(moeri,e,nocc)
E = ehf + mp2_corr
println(E)
println("moeri homebrew")
println(moeri)
#println("moeri homebrew 2")
#println(moeri2)
println("moeri mints")
println(moeri1)
println(length(e))

l
