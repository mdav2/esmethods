module Determinant
"Module for creating and working with Slater determinants."

using Wavefunction
export SlaterDeterminant
export norbdiff
export orbdiff

struct SlaterDeterminant
	alpha::BitArray
	beta::BitArray
end
function norbdiff(a::SlaterDeterminant,b::SlaterDeterminant)
	return reduce(+,reduce(+,orbdiff(a,b)))
end
function orbdiff(a::SlaterDeterminant,b::SlaterDeterminant)
	return a.alpha .⊻ b.alpha, a.beta .⊻ b.beta
end
end #module
