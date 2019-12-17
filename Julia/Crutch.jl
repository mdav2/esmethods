#a couple of functions to convert wfn objects to pure
#arrays.
using PyCall
psi4 = pyimport("psi4")
function init(mol,basis)
    psi4.core.be_quiet()
    psi4.set_options(Dict("scf_type" => "pk",
    		              "basis" => basis))
    ehf,wfn = psi4.energy("hf",mol=mol,return_wfn=true)
    return wfn
end
function std_restricted(wfn)
	return std_restricted(wfn,Float64)
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
function setup_rccd(wfn,dt)
	nso,nbf,nocc,noccso,nvir,nvirso,eps,mints,_C = std_restricted(wfn,dt)
	mo_eri = convert(Array{dt},mints.mo_eri(_C,_C,_C,_C).to_array())
	Wmnij = zeros(dt,nocc,nocc,nocc,nocc)
	Wabef = zeros(dt,nvir,nvir,nvir,nvir)
	WmBeJ = zeros(dt,nocc,nvir,nvir,nocc)
	WmBEj = zeros(dt,nocc,nvir,nvir,nocc)
	tiJaB = zeros(dt,nocc,nocc,nvir,nvir)
	return permutedims(mo_eri,[1,3,2,4]),tiJaB,Wmnij,Wabef,WmBeJ,WmBEj,eps
end
function setup_rccd(wfn)
	return setup_rccd(wfn,Float64)
end
