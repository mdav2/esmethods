module CC
export form_Wmnij
export form_Wabef
export form_WmBeJ
export form_WmBEj
export form_T2!
export form_Dijab
export ccenergy
export cciter
export T2_init!
function rccd(nso,nbf,nocc)
end

function T2_init!(tiJaB,iJaB,Dijab)
	nocc = size(tiJaB,1)
	nvir = size(tiJaB,4)
	@views iJaB_oovv = iJaB[1:nocc,1:nocc,nocc+1:nocc+nvir,nocc+1:nocc+nvir]
	rocc = collect(UnitRange(1,nocc))
	rvir = collect(UnitRange(1,nvir))
	for i in rocc
		for j in rocc
			for a in rvir
				for b in rvir
					tiJaB[i,j,a,b] = (iJaB_oovv[i,j,a,b])/Dijab[i,j,a,b]
				end
			end
		end
	end
end

function form_Dijab(tiJaB,F)
	nocc = size(tiJaB,1)
	nvir = size(tiJaB,4)
	rocc = collect(UnitRange(1,nocc))
	rvir = collect(UnitRange(1,nvir))
	Dijab = zeros(Float64,nocc,nocc,nvir,nvir)
	for i in rocc
		for j in rocc
			for a in rvir
				for b in rvir
					aa = a + nocc
					bb = b + nocc
					Dijab[i,j,a,b] = F[i] + F[j] - F[aa] - F[bb]
				end
			end
		end
	end
	return Dijab
end
function form_T2(tiJaB_i,WmBeJ,WmBEj,Wabef,Wmnij,iJaB,Dijab)
	nocc = size(Wmnij,1)
	nvir = size(tiJaB_i,4)
	tiJaB_d = zeros(Float64,nocc,nocc,nvir,nvir)
	@views iJaB_oovv = iJaB[1:nocc,1:nocc,nocc+1:nocc+nvir,nocc+1:nocc+nvir]
	rocc = collect(UnitRange(1,nocc))
	rvir = collect(UnitRange(1,nvir))
	tiJaB_d += iJaB_oovv
	for i in rocc
		for j in rocc
			for a in rvir
				for b in rvir
					#term 6
					for m in rocc
						for n in rocc
							tiJaB_d[i,j,a,b] += tiJaB_i[m,n,a,b]*Wmnij[m,n,i,j]
						end
					end
					#term 7
					for e in rvir
						for f in rvir
							tiJaB_d[i,j,a,b] += tiJaB_i[i,j,e,f]*Wabef[a,b,e,f]
						end
					end
					#term 8,9,10,11,12,13
					for e in rvir
						for m in rocc
							#8
							tiJaB_d[i,j,a,b] += (tiJaB_i[i,m,a,e] - tiJaB_i[m,i,a,e])*WmBeJ[m,b,e,j]
							#9
							tiJaB_d[i,j,a,b] += tiJaB_i[i,m,a,e]*(WmBeJ[m,b,e,j] + WmBEj[m,b,e,j])
							#10
							tiJaB_d[i,j,a,b] += tiJaB_i[m,i,b,e]*WmBEj[m,a,e,j]
							#11
							tiJaB_d[i,j,a,b] += tiJaB_i[m,j,a,e]*WmBEj[m,b,e,i]
							#12
							tiJaB_d[i,j,a,b] += (tiJaB_i[j,m,b,e] - tiJaB_i[m,j,b,e])*WmBeJ[m,a,e,i]
							#13
							tiJaB_d[i,j,a,b] += tiJaB_i[j,m,b,e]*WmBeJ[m,a,e,i]
							tiJaB_d[i,j,a,b] += tiJaB_i[j,m,b,e]*WmBEj[m,a,e,i]
						end
					end
					tiJaB_d[i,j,a,b] = tiJaB_d[i,j,a,b]/Dijab[i,j,a,b]
				end
			end
		end
	end
	return tiJaB_d
end

function form_Wmnij(iJaB::Array{Float64,4},tiJaB::Array{Float64,4})
	#Wmnij for RCCD
	#fills Wmnij (!)	
	nocc = size(tiJaB,1)
	nvir = size(tiJaB,4)
	Wmnij = zeros(Float64,nocc,nocc,nocc,nocc)
	@views iJaB_oovv = iJaB[1:nocc,1:nocc,nocc+1:nocc+nvir,nocc+1:nocc+nvir]
	rocc = collect(UnitRange(1,nocc))
	rvir = collect(UnitRange(1,nvir))

	Wmnij += iJaB[1:nocc,1:nocc,1:nocc,1:nocc]
	for m in rocc
		for n in rocc
			for i in rocc
				for j in rocc
					#start W[m,n,i,j]
					for e in rvir
						for f in rvir
							Wmnij[m,n,i,j] += tiJaB[i,j,e,f]*iJaB_oovv[m,n,e,f]/2
						end
					end
				end
			end
		end
	end
	return Wmnij
end

function form_Wabef(iJaB::Array{Float64,4},tiJaB::Array{Float64})
	nocc = size(tiJaB,1)
	nvir = size(tiJaB,4)
	Wabef = zeros(Float64,nvir,nvir,nvir,nvir)
	Wabef += iJaB[nocc+1:nvir+nocc,nocc+1:nvir+nocc,nocc+1:nvir+nocc,nocc+1:nvir+nocc]
	rocc = collect(UnitRange(1,nocc))
	rvir = collect(UnitRange(1,nvir))
	@views iJaB_oovv = iJaB[1:nocc,1:nocc,nocc+1:nocc+nvir,nocc+1:nocc+nvir]
	for a in rvir
		for b in rvir
			for e in rvir
				for f in rvir
					for n in rocc
						for m in rocc
							Wabef[a,b,e,f] += tiJaB[m,n,a,b]*iJaB_oovv[m,n,e,f]/2
						end
					end
				end
			end
		end
	end
	return Wabef
end

function form_WmBeJ(iJaB, tiJaB)
	nocc = size(tiJaB,1)
	nvir = size(tiJaB,4)
	WmBeJ = zeros(Float64,nocc,nvir,nvir,nocc)
	WmBeJ += iJaB[1:nocc,nocc+1:nocc+nvir,nocc+1:nocc+nvir,1:nocc]
	rocc = collect(UnitRange(1,nocc))
	rvir = collect(UnitRange(1,nvir))
	@views iJaB_oovv = iJaB[1:nocc,1:nocc,nocc+1:nocc+nvir,nocc+1:nocc+nvir]
	for m in rocc
		for b in rvir
			for e in rvir
				for j in rocc
					for f in rvir
						for n in rocc
							WmBeJ[m,b,e,j] -= tiJaB[j,n,f,b]*iJaB_oovv[m,n,e,f]/2
							WmBeJ[m,b,e,j] += tiJaB[n,j,f,b]*iJaB_oovv[m,n,e,f]
							WmBeJ[m,b,e,j] -= tiJaB[n,j,f,b]*iJaB_oovv[n,m,e,f]/2
						end
					end
				end
			end
		end
	end
	return WmBeJ
end

function form_WmBEj(iJaB,tiJaB)
	nocc = size(tiJaB,1)
	nvir = size(tiJaB,4)
	WmBEj = zeros(Float64,nocc,nvir,nvir,nocc)
	rocc = collect(UnitRange(1,nocc))
	rvir = collect(UnitRange(1,nvir))
	iJaB_2134 = permutedims(iJaB,[2,1,3,4])
	@views iJaB_ovvo = iJaB_2134[1:nocc,nocc+1:nocc+nvir,nocc+1:nocc+nvir,1:nocc]
	@views iJaB_oovv = iJaB[1:nocc,1:nocc,nocc+1:nocc+nvir,nocc+1:nocc+nvir]
	WmBEj -= iJaB_ovvo
	for m in rocc
		for b in rvir
			for e in rvir
				for j in rocc
					for n in rocc
						for f in rvir
							WmBEj[m,b,e,j] += tiJaB[j,n,f,b]*iJaB_oovv[n,m,e,f]/2
						end
					end
				end
			end
		end
	end
	return WmBEj
end
function ccenergy(tiJaB,iJaB)
	ecc = 0
	nocc = size(tiJaB,1)
	nvir = size(tiJaB,4)
	rocc = collect(UnitRange(1,nocc))
	rvir = collect(UnitRange(1,nvir))
	@views iJaB_oovv = iJaB[1:nocc,1:nocc,nocc+1:nocc+nvir,nocc+1:nocc+nvir]
	for i in rocc
		for j in rocc
			for a in rvir
				for b in rvir
					ecc += iJaB_oovv[i,j,a,b]*2*tiJaB[i,j,a,b]
					ecc -= iJaB_oovv[i,j,a,b]*tiJaB[j,i,a,b]
				end
			end
		end
	end
	return ecc
end
function cciter(tiJaB_i,iJaB,Dijab)
	Wmnij = form_Wmnij(iJaB,tiJaB_i)
	Wabef = form_Wabef(iJaB,tiJaB_i)
	WmBeJ = form_WmBeJ(iJaB,tiJaB_i)
	WmBEj = form_WmBEj(iJaB,tiJaB_i)
	tiJaB_d = form_T2(tiJaB_i,WmBeJ,WmBEj,Wabef,Wmnij,iJaB,Dijab)
	return tiJaB_d
end
end #module CC
