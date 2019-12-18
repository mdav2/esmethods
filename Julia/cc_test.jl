using PyCall
#include("CoupledCluster.jl")
using CoupledCluster
psi4 = pyimport("psi4")
include("Crutch.jl")

mol = psi4.geometry("""
		    O
		    H 1 1.1
		    H 1 1.1 2 104.0
		    symmetry c1
		    """)
psi4.set_options(Dict("basis" => "sto-3g", "scf_type" => "pk",
					  "d_convergence" => 14))
wfn = init(mol,"cc-pvdz")
#println(wfn.current_energy())
refWfn = PyToJl(wfn,Float64,false)
do_rccd(refWfn)
