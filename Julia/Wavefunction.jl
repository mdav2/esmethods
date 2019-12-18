module Wavefunction
export Wfn
mutable struct Wfn{T}
    #wfn class for SCF/reference wavefunctions
	nalpha::Int
	nbeta::Int
    nvira::Int
    nvirb::Int
	nmo::Int
	unrestricted::Bool
	#kint::Array{T,2}
	#pint::Array{T,2}
    Ca::Array{T,2} #AO->MO coefficients
    Cb::Array{T,2} #AO->MO coefficients
    ha::Array{T,2} #Core hamiltonian
    hb::Array{T,2} #Core hamiltonian
    epsa::Array{T,1} #orbital eigenvalues
    epsb::Array{T,1} #orbital eigenvalues
    uvsr::Array{T,4} #AO basis electron repulsion integrals

    pqrs::Array{T,4} #MO basis electron repulsion integrals
    pQrS::Array{T,4} #MO basis electron repulsion integrals
    pQRs::Array{T,4} #MO basis electron repulsion integrals
    PQRS::Array{T,4} #MO basis electron repulsion integrals
    PqRs::Array{T,4} #MO basis electron repulsion integrals
    PqrS::Array{T,4} #MO basis electron repulsion integrals
end
function Wfn{T}(a::Int) where {T}
    l = Array{T}(undef,0)
	m = Array{T}(undef,0,0)
	n = Array{T}(undef,0,0,0,0)
	Wfn{T}(a,a,0,true,
           m,m,m,m, #2D
           l,l,
           n,n,n,n,n,n,n) #4D
end
end
