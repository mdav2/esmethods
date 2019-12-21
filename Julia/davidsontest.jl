using Davidson
using LinearAlgebra

n = 10
A = zeros(n,n)
A += I(n)
for i in UnitRange(1,n)
	A[i,i] = i
end
#println(A)

#A += transpose(A)
#A ./ 2.0
A = Symmetric(A)

eigdav(A,1,4,40,1*10^-6)
val,vec = eigen(A)
