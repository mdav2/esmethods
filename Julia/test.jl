include("Wavefunction.jl")
include("Crutch.jl")
using .Wavefunction
using PyCall
psi4 = pyimport("psi4")
psi4.core.be_quiet()

mol = psi4.geometry("""
                    O
                    H 1 1.1
                    H 1 1.1 2 104.0
                    symmetry c1
                    """)
psi4.set_options(Dict("basis" => "sto-3g", 
                      "scf_type" => "pk"))
e,wfn = psi4.energy("scf",mol=mol,return_wfn=true)
mywfn = PyToJl(wfn,Float64,false)
println(mywfn.Ca)
