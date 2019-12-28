__precompile__()
module Wavefunction
"Module for storing and handling reference wavefunctions"

using PyCall
#psi4 = pyimport("psi4")
const psi4 = PyNULL()
function __init__()
	copy!(psi4,pyimport("psi4"))
end
export init
export Wfn
export PyToJl
struct Wfn{T}
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
struct directWfn{T}
	#requires the module using this struct to have properly imported
	#psi4 with pycall
	nalpha::Int
	nbeta::Int
    nvira::Int
    nvirb::Int
	nmo::Int
	unrestricted::Bool
    Ca::Array{T,2} #AO->MO coefficients
    Cb::Array{T,2} #AO->MO coefficients
    ha::Array{T,2} #Core hamiltonian
    hb::Array{T,2} #Core hamiltonian
    epsa::Array{T,1} #orbital eigenvalues
    epsb::Array{T,1} #orbital eigenvalues
	basis::PyObject
	mints::PyObject
end
function PyToJl(wfn,dt,unrestricted::Bool)
	"Takes in a psi4 wavefunction and converts important information
	to pure Julia objects for stability."
    dummy2 = Array{dt}(undef,0,0) #placeholder 2D array
    dummy4 = Array{dt}(undef,0,0,0,0) #placeholder 4D array
    _Ca   = wfn.Ca() #as psi4 Matrix objects for use with MintsHelper
    _Cb   = wfn.Cb() #as psi4 Matrix objects for use with MintsHelper
    nbf   = wfn.nmo()
    nocca =  wfn.nalpha()
    nvira = nbf - nocca
    mints = psi4.core.MintsHelper(wfn.basisset()) #to generate integrals
    epsa  = convert(Array{dt,1},wfn.epsilon_a().to_array()) #orbital eigenvalues
    Ca    = convert(Array{Float64,2}, _Ca.to_array())
    hao   = convert(Array{Float64,2}, wfn.H().to_array()) #core hamiltonian in AO
    Cb    = convert(Array{dt,2},_Cb.to_array())
    noccb = wfn.nbeta()
    nvirb = nbf - noccb
    epsb  = convert(Array{dt,1},wfn.epsilon_b().to_array()) #orbital eigenvalues
    uvsr  = convert(Array{dt,4},mints.ao_eri().to_array()) #AO basis integrals
    pqrs  = convert(Array{dt,4},mints.mo_eri(_Ca,_Ca,_Ca,_Ca).to_array()) #MO basis integrals
    if unrestricted #avoid making these if not an unrestricted or open shell wfn
		#various spin cases notation --> alpha BETA
        pQrS  = convert(Array{dt,4},mints.mo_eri(_Ca,_Cb,_Ca,_Cb).to_array())
        pQRs  = convert(Array{dt,4},mints.mo_eri(_Ca,_Cb,_Cb,_Ca).to_array())
        PQRS  = convert(Array{dt,4},mints.mo_eri(_Cb,_Cb,_Cb,_Cb).to_array())
        PqRs  = convert(Array{dt,4},mints.mo_eri(_Cb,_Ca,_Cb,_Ca).to_array())
        PqrS  = convert(Array{dt,4},mints.mo_eri(_Cb,_Ca,_Ca,_Cb).to_array())
    else
		#just fill with placeholder for RHF case
        pQrS  = dummy4
        pQRs  = dummy4
        PQRS  = dummy4
        PqRs  = dummy4
        PqrS  = dummy4
    end
	#create the Wfn object and return it!
    owfn = Wfn{dt}(nocca,noccb,nvira,nvirb,nbf,unrestricted,
           Ca,Cb,dummy2,dummy2,
           epsa,epsb,
           uvsr,pqrs,pQrS,pQRs,PQRS,PqRs,PqrS)
    return owfn
end
function init(mol)
	"Initialize molecule object to a HF wfn for reference"
    ehf,wfn = psi4.energy("hf",mol=mol,return_wfn=true)
    return wfn
end
end
