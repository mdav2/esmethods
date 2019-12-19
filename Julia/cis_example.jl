#ENV["PYTHON"] = "/home/matthewmcallisterdavis/miniconda3/envs/psi/bin/python"
#import Pkg
#Pkg.build("PyCall")
using PyCall
using Wavefunction
using CISingles
psi4 = pyimport("psi4")
psi4.core.be_quiet()
#include("Crutch.jl")

mol = psi4.geometry("""
		    O
		    H 1 1.1
		    H 1 1.1 2 104.0
		    symmetry c1
		    """)
psi4.set_options(Dict("basis" => "sto-3g", "scf_type" => "pk"))
wfn = init(mol)
nocc,nvir,mo_eri,F = setup_rcis(wfn,Float64)
println(do_CIS(nocc,nvir,mo_eri,F,2))
