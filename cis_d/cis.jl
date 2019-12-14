using PyCall
using BenchmarkTools
psi4 = pyimport("psi4")
np = pyimport("numpy")
mol = psi4.geometry("""
    O
    H 1 1.1
    H 1 1.1 2 104.0
    symmetry c1""")
psi4.core.be_quiet()
basis = "sto-3g"
psi4.set_options(Dict("scf_type" => "pk",
		      "basis" => basis))
ehf,wfn = psi4.energy("hf",mol=mol,return_wfn=true)
