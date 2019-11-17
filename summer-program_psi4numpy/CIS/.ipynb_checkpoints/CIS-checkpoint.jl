"""
A julia program to compute configuration interaction singles
    excitation energies.
About 200 LOC. 
Uses full diagonalization of the hamiltonian matrix, and hence scales poorly.
#TODO: implement a davidson solver!
"""

"""
Setup of libraries that are needed
"""

using PyCall
using LinearAlgebra
using BenchmarkTools
determinant = pyimport("helper_CI").Determinant
hamiltonianGenerator = pyimport("helper_CI").HamiltonianGenerator
psi4 = pyimport("psi4")
np = pyimport("numpy")

function setup_cis(wfn)
    """
    This function extracts the information required to do
    CIS from a wavefunction.
    inputs:
        wfn::psi4.Wavefunction object
    outputs:
        C::Array{Float64,2} -> coefficient matrix for AO-MO conversion
        ndocc::Int64 -> number of doubly occupied orbitals
        nmo::Int64 -> number of molecular orbitals
        H::Array{Float64,2} -> core potential + kinetic integrals in AO basis
        MO::Array{Float64,4} -> two electron electron repulsion integrals
    """

    mints = psi4.core.MintsHelper(wfn.basisset())
    C = wfn.Ca()
    ndocc = wfn.doccpi()[1]
    nmo = wfn.nmo()
    #nvirt = (nmo - ndocc)
    H = np.asarray(mints.ao_kinetic()) + np.asarray(mints.ao_potential())
    MO = np.asarray(mints.mo_spin_eri(C,C))
    return C.to_array(),ndocc,nmo,H,MO
end


function transform_H(C,H)
    """
    Converts a core hamiltonian matrix from AO basis to MO basis.
    Uses np.einsum for transformation ('impure' Julia)

    inputs:
        C::Array{Float64,2} -> AO-MO coefficients
        H::Array{Float64,2} -> core hamiltonian matrix in AO basis.
    outputs:
        H_o::Array{Float64,2} -> core hamiltonian matrix in MO basis
    """

    L = length(C[1,:])*2
    outh = zeros(L,L)
    H_o = np.einsum("uj,vi,uv",C,C,H,optimize=false)
    H_o = np.repeat(H_o,2,axis=0)
    H_o = np.repeat(H_o,2,axis=1)
    spin_ind = collect(1:1:L)
    for (idx,val) in enumerate(spin_ind)
        spin_ind[idx] = val%2
    end
    truth = zeros(L,L)
    for i in 1:1:length(spin_ind)
        for j in 1:1:length(spin_ind)
            truth[i,j] = (spin_ind[i] == transpose(spin_ind[j]))
        end
    end
    for (idx,val) in enumerate(spin_ind)
        spin_ind[idx] = val%2
    end
    for i in 1:1:L
        for j in 1:1:L
            H_o[i,j] *= truth[i,j]
        end
    end
    return H_o
end

function transform_H_alt(C::Array{Float64,2},H::Array{Float64})
    """
    Converts a core hamiltonian matrix from AO basis to MO basis.
    Uses for loops for transformation ('pure' Julia)

    inputs:
        C::Array{Float64,2} -> AO-MO coefficients
        H::Array{Float64,2} -> core hamiltonian matrix in AO basis.
    outputs:
        H_o::Array{Float64,2} -> core hamiltonian matrix in MO basis
    """

    L = length(C[1,:])
    H_o = zeros(L, L)
    @inbounds for j in 1:1:L
        @inbounds for i in 1:1:L
            @inbounds for u in 1:1:L
                @inbounds for v in 1:1:L
                    @fastmath H_o[i,j] += C[u,j]*C[v,i]*H[u,v]
                end
            end
        end
    end

    H_o = np.repeat(H_o,2,axis=0)
    H_o = np.repeat(H_o,2,axis=1)

    spin_ind = collect(1:1:L*2)
    for (idx,val) in enumerate(spin_ind)
        spin_ind[idx] = val%2
    end
    truth = zeros(L*2,L*2)
    spin_ind_t = transpose(spin_ind)
    for i in 1:1:length(spin_ind)
        for j in 1:1:length(spin_ind)
            truth[i,j] = (spin_ind[i] == spin_ind_t[j])
            #truth[i,j] = (spin_ind[i] == transpose(spin_ind[j]))
        end
    end
    for (idx,val) in enumerate(spin_ind)
        spin_ind[idx] = val%2
    end
    for i in 1:1:L*2
        for j in 1:1:L*2
            H_o[i,j] *= truth[i,j]
        end
    end
    return H_o
end
function davidson(matrix,_eigs,k,kmax,tol)
    eigs = _eigs
    l = ones(eigs)
    n = length(matrix[:,1])
    #k = 4
    t = Array(I(n))#::Array{Float64,2}
    V = zeros((n,n))::Array{Float64,2}

   # kmax = 100
   # tol = 1E-14::Float64
    w = zeros(n)
    temp1 = zeros((n,))
    for j in 1:1:k
        V[:,j] = t[:,j]
    end
    l = ones(eigs)::Array{Float64,1}
    normm = 0
    @inbounds for m in k:k:kmax
        #println(m)
        l_old = l[1:eigs]
        @fastmath qr!(V)
        #V = Array(V.Q)::Array{Float64,2}
        #Q = Array(QR.Q)::Array{Float64,2}#Array{Float64,2}
        @fastmath T = transpose(V[1:n,1:(m+1)])*A*V[1:n,1:(m+1)]::Array{Float64,2}
        @fastmath L = eigvals(T)::Array{Float64}
        @fastmath E = eigvecs(T)::Array{Float64}
        idx = sortperm(L)
        l = L[idx]
        e = E[:,idx]
        w .= 0
        @inbounds for j in 1:1:k
            @fastmath V[1:n,j+m+1] = (A - l[j]*t)*(V[1:n,1:(m+1)]*e[:,j])/(l[j] - A[j,j])
            #@fastmath w *= 1/(l[j] - A[j,j])
             # w
        end
        
        @fastmath if norm(l[1:eigs] - l_old)::Float64 < tol
            break
        end
    end
    return l[1:eigs]#normm
end

function build_DetList(ndocc,nmo)
    """
    Constructs a list of singly excited determinants
    from an occupation array and total number of MOs.
    inputs:
        ndocc::Int64 -> number of doubly occupied orbitals
        nmo::Int64 -> total number of molecular orbitals
    outputs:
        detList::Array{determinant::PyObject} -> array of singly excited
                                                 determinants.
    """
    occList = collect(0:1:(ndocc-1))
    det_ref = determinant(alphaObtList=occList,betaObtList=occList)
    detList = det_ref.generateSingleExcitationsOfDet(nmo)::Array{PyObject}
    append!(detList,[det_ref])
    return detList
end

function build_HamiltonianMatrix(H, MO, detList)
    """
    Constructs the hamiltonian matrix from a core hamiltonian matrix,
    electron repulsion integrals, and a list of determinants.
    inputs:
        H::Array{Float64,2} -> core hamiltonian matrix in MO basis
        MO::Array{Float64,4} -> tensor of electron repulsion integrals
        detList::Array{determinant::PyObject} -> determinants (ref + excited)
                                                 to be considered.
    outputs:
        Hamiltonian_matrix::Array{Float64,2} -> matrix elements between singly
                                                excited determinants.
    """
    Hamiltonian_generator = hamiltonianGenerator(H,MO)
    Hamiltonian_matrix = Hamiltonian_generator.generateMatrix(detList)
    return Hamiltonian_matrix
end

function cis(wfn)
    """
    Computes the CIS excitation energies for an input (RHF) wavefunction.
    inputs:
        wfn::psi4.Wavefunction::PyObject -> wavefunction resulting from an RHF
                                            SCF computation in psi4.
    outputs:
        evals::Array{Float64,1} -> CIS excitation energies of the input wfn.
                                   Includes ground state (0 eigenvalue).
    """
    C,ndocc,nmo,H,MO = setup_cis(wfn)
    detList = build_DetList(ndocc,nmo)
    H = transform_H(C,H)
    H_matrix = build_HamiltonianMatrix(H,MO,detList)
    #evals = eigvals(H_matrix)
    evals = davidson(H_matrix,1,4,40,1E-8)
    for (idx,val) in enumerate(evals)
        evals[idx] += mol.nuclear_repulsion_energy()
        evals[idx] -= scf_e
    end
    return evals
end



"""Build molecule, set up psi4, and compute HF wavefunction."""
mol = psi4.geometry("""
    O
    H 1 0.96
    H 1 0.96 2 104.5
    symmetry c1""")
psi4.core.be_quiet()
psi4.core.clean()
psi4.set_memory("2GB")
psi4.set_options(Dict("scf_type" => "pk",
                      "reference" => "rhf",
                      "basis" => "cc-pvdz"))
scf_e,wfn = psi4.energy("scf",return_wfn=true)
cis_energies = cis(wfn)
println(cis_energies)
