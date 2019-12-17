using PyCall
dt = Float64
psi4 = pyimport("psi4")
function init(mol,basis)
    psi4.core.be_quiet()
    psi4.set_options(Dict("scf_type" => "pk",
    		              "basis" => basis))
    ehf,wfn = psi4.energy("hf",mol=mol,return_wfn=true)
    return wfn
end
function setup_cis(wfn)
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
