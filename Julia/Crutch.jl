# a couple of functions to convert wfn objects to pure
# arrays and set up some computations.
# TODO: strip this file to just init and PyToJl functions
using PyCall
using Wavefunction
psi4 = pyimport("psi4")
function init(mol)
	"Initialize molecule object to a HF wfn for reference"
    ehf,wfn = psi4.energy("hf",mol=mol,return_wfn=true)
    return wfn
end
function std_restricted(wfn)
    "Old function, similar to PyToJl"	
	return std_restricted(wfn,Float64)
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
function std_restricted(wfn,dt)
    _C = wfn.Ca()
	C = _C.to_array()
    nbf = wfn.nmo()
    nso = 2*nbf
    noccso= 2*wfn.nalpha()
    nvirso = nso - noccso
	nocc = wfn.nalpha()
	nvir = nbf - nocc
    eps = wfn.epsilon_a().to_array()
    basis = wfn.basisset()
    mints = psi4.core.MintsHelper(basis)
	return nso,nbf,nocc,noccso,nvir,nvirso,eps,mints,_C
end
function setup_rcis(wfn,dt)
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
function setup_rcis(wfn)
    return setup_rcis(wfn,Float64)
end
