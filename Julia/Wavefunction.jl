module Wavefunction
export wfn
mutable struct wfn{T}
	nalpha::Int
	nbeta::Int
	nmo::Int
	unrestricted::Bool
	kint::Array{T,2}
	pint::Array{T,2}
	teint::Array{T,4}
end
function wfn{T}(a::Int) where {T}
	m = Array{T}(undef,0,0)
	n = Array{T}(undef,0,0,0,0)
	wfn{T}(a,a,0,true,m,m,n)
end
end
