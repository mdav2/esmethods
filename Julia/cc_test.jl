using PyCall
using CC
psi4 = pyimport("psi4")
include("Crutch.jl")

mol = psi4.geometry("""
		    O
		    H 1 1.1
		    H 1 1.1 2 104.0
		    symmetry c1
		    """)
psi4.set_options(Dict("basis" => "sto-3g", "scf_type" => "pk",
					  "d_convergence" => 10))
wfn = init(mol,"sto-3g")
iJaB,tiJaB,Wmnij,Wabef,WmBeJ,WmBEj,eps = setup_rccd(wfn)
println("CCD set up")
println("formed Wmnij")
println("formed Wabef")
println("formed WmBeJ")
println("formed WmBEj")
Dijab = form_Dijab(tiJaB, eps)
println("formed Dijab")
nocc = size(tiJaB,1)
nvir = size(tiJaB,4)
T2_init!(tiJaB,iJaB,Dijab)
T2_d = zeros(Float64,nocc,nocc,nvir,nvir)
#form_T2!(tiJaB_new,tiJaB,WmBeJ,WmBEj,Wabef,Wmnij,iJaB,Dijab)
#println(size(tiJaB))
println(ccenergy(tiJaB,iJaB))
T2_d = cciter(tiJaB,iJaB,Dijab)
println(T2_d)
println(ccenergy(T2_d,iJaB))
