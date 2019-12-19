module CoupledCluster
using Wavefunction
using LinearAlgebra
using Dates
export do_rccd
export form_Wmnij
export form_Wabef
export form_WmBeJ
export form_WmBEj
export form_T2!
export form_Dijab
export ccenergy
export cciter
export T2_init!


function do_rccd(refWfn::Wfn)
    #Implicit dtype = float64
    return do_rccd(refWfn,Float64,40)
end
function do_rccd(refWfn::Wfn,dt,maxit)
    #goes through appropriate steps to do RCCD
    nocc = refWfn.nalpha
    nvir = refWfn.nvira
    iJaB = permutedims(refWfn.pqrs,[1,3,2,4])
    epsa = refWfn.epsa
	T2 = zeros(dt,nocc,nocc,nvir,nvir)
    Dijab = form_Dijab(T2,epsa)
    T2_init!(T2,iJaB,Dijab)
    println(ccenergy(T2,iJaB))
    Fae = form_Fae(T2,iJaB)
    Fmi = form_Fmi(T2,iJaB)
    Wmnij = form_Wmnij(iJaB,T2)
	Wabef = form_Wabef(iJaB,T2)
	WmBeJ = form_WmBeJ(iJaB,T2)
	WmBEj = form_WmBEj(iJaB,T2)
    for i in UnitRange(1,maxit)
        t0 = Dates.Time(Dates.now())
        T2 = cciter(T2,iJaB,Dijab,Fae,Fmi,Wabef,Wmnij,WmBeJ,WmBEj)
        t1 = Dates.Time(Dates.now())
        print("T2 formed in ")
        print(convert(Dates.Millisecond, (t1 - t0)))
        print("\n")
        t0 = Dates.Time(Dates.now())
        print("@CCD ")
        print(ccenergy(T2,iJaB))
        print
        t1 = Dates.Time(Dates.now())
        print("\n")
        print("energy computed in ")
        print(convert(Dates.Millisecond, (t1 - t0)))
        print("\n")
    end
    println(ccenergy(T2,iJaB))
end
function ccenergy(tiJaB,iJaB)
	ecc = 0.0
	nocc = size(tiJaB,1)
	nvir = size(tiJaB,4)
	rocc = collect(UnitRange(1,nocc))
	rvir = collect(UnitRange(1,nvir))
	@views iJaB_oovv = iJaB[1:nocc,1:nocc,nocc+1:nocc+nvir,nocc+1:nocc+nvir]
	for i in rocc
		for j in rocc
			for a in rvir
				for b in rvir
                    cache = iJaB_oovv[i,j,a,b]
					#ecc += iJaB_oovv[i,j,a,b]*2*tiJaB[i,j,a,b]
					ecc += cache*2*tiJaB[i,j,a,b]
					ecc -= cache*tiJaB[j,i,a,b]
				end
			end
		end
	end
	return ecc
end

function cciter(tiJaB_i,iJaB,Dijab,Fae,Fmi,Wabef,Wmnij,WmBeJ,WmBEj)
    form_Fae!(Fae,tiJaB_i,iJaB)
    Fmi = form_Fmi!(Fmi,tiJaB_i,iJaB)
    Wmnij = form_Wmnij!(Wmnij,iJaB,tiJaB_i)
	Wabef = form_Wabef!(Wabef,iJaB,tiJaB_i)
	WmBeJ = form_WmBeJ!(WmBeJ,iJaB,tiJaB_i)
	WmBEj = form_WmBEj!(WmBEj,iJaB,tiJaB_i)
	tiJaB_d = form_T2(tiJaB_i,Fae,Fmi,WmBeJ,WmBEj,Wabef,Wmnij,iJaB,Dijab)
	return tiJaB_d
end

function T2_init!(tiJaB,iJaB,Dijab)
	nocc = size(tiJaB,1)
	nvir = size(tiJaB,4)
	@views iJaB_oovv = iJaB[1:nocc,1:nocc,nocc+1:nocc+nvir,nocc+1:nocc+nvir]
	rocc = collect(UnitRange(1,nocc))
	rvir = collect(UnitRange(1,nvir))
    tiJaB .= iJaB_oovv ./ Dijab
	#for i in rocc
	#	for j in rocc
	#		for a in rvir
	#			for b in rvir
	#				tiJaB[i,j,a,b] = (iJaB_oovv[i,j,a,b])/Dijab[i,j,a,b]
	#			end
	#		end
	#	end
	#end
end

function form_Fae(tiJaB,iJaB)
	nvir = size(tiJaB,4)
    Fae = zeros(Float64,nvir,nvir)
    form_Fae!(Fae,tiJaB,iJaB)
    return Fae
end
function form_Fae!(Fae,tiJaB,iJaB)
	nocc = size(tiJaB,1)
	nvir = size(tiJaB,4)
	rocc = collect(UnitRange(1,nocc))
	rvir = collect(UnitRange(1,nvir))
	@views iJaB_oovv = iJaB[1:nocc,1:nocc,nocc+1:nocc+nvir,nocc+1:nocc+nvir]
    Fae .= 0.0
    for a in rvir
        for e in rvir
            for f in rvir
                for n in rocc
                    for m in rocc
                        Fae[a,e] -= tiJaB[m,n,a,f]*(2*iJaB_oovv[m,n,e,f] - iJaB_oovv[n,m,e,f])
                    end
                end
            end
        end
    end
end

function form_Fmi(tiJaB,iJaB)
	nocc = size(tiJaB,1)
	nvir = size(tiJaB,4)
    Fmi = zeros(Float64,nocc,nocc)
    form_Fmi!(Fmi,tiJaB,iJaB)
end
function form_Fmi!(Fmi,tiJaB,iJaB)
	nocc = size(tiJaB,1)
	nvir = size(tiJaB,4)
	rocc = collect(UnitRange(1,nocc))
	rvir = collect(UnitRange(1,nvir))
    Fmi .= 0.0
	@views iJaB_oovv = iJaB[1:nocc,1:nocc,nocc+1:nocc+nvir,nocc+1:nocc+nvir]
    for f in rvir
        for e in rvir
            for n in rocc
                for i in rocc
                    for m in rocc
                        Fmi[m,i] += tiJaB[i,n,e,f]*(2*iJaB_oovv[m,n,e,f] - iJaB_oovv[m,n,f,e])
                    end
                end
            end
        end
    end
    return Fmi
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

@fastmath @inbounds function form_T2(tiJaB_i,Fae,Fmi,WmBeJ,WmBEj,Wabef,Wmnij,iJaB,Dijab)
	nocc = size(Wmnij,1)
	nvir = size(tiJaB_i,4)
	tiJaB_d = zeros(Float64,nocc,nocc,nvir,nvir)
	@views iJaB_oovv = iJaB[1:nocc,1:nocc,nocc+1:nocc+nvir,nocc+1:nocc+nvir]
	rocc = collect(UnitRange(1,nocc))
	rvir = collect(UnitRange(1,nvir))
	#tiJaB_d .= iJaB_oovv
    _Wabef = zeros(Float64,nvir,nvir)
	for b in rvir
        for a in rvir #collect(UnitRange(b+1,nvir))
            _Wabef .= Wabef[a,b,:,:]
		    for j in rocc
                for i in collect(UnitRange(j+1,nocc))
                    #term 2
                    temp = iJaB_oovv[i,j,a,b]
                    #@views T2_ija = tiJaB_i[i,j,a,:]
                    @views temp += BLAS.dot(tiJaB_i[i,j,a,:] , Fae[b,:])#reduce(+, tiJaB_i[i,j,a,:] .* Fae[b,:] )
                    @views temp += BLAS.dot(tiJaB_i[i,j,:,b], Fae[a,:])#reduce(+, tiJaB_i[i,j,:,b] .* Fae[a,:] )

					for m in rocc
                        @views temp += BLAS.dot(tiJaB_i[m,i,b,:],WmBEj[m,a,:,j])
                        @views temp += BLAS.dot(tiJaB_i[m,j,a,:],WmBEj[m,b,:,i])
                        for e in rvir
							#8
							temp += (tiJaB_i[i,m,a,e] - tiJaB_i[m,i,a,e])*WmBeJ[m,b,e,j]
							#9
							temp += tiJaB_i[i,m,a,e]*(WmBeJ[m,b,e,j] + WmBEj[m,b,e,j])
							#10
                            cache = tiJaB_i[j,m,b,e]
							##13
							temp += (cache - tiJaB_i[m,j,b,e])*WmBeJ[m,a,e,i]
							#13
                            temp += cache*(WmBeJ[m,a,e,i] + WmBEj[m,a,e,i])
						end
                    end
                    @views temp += sum( tiJaB_i[i,j,:,:] .* _Wabef[:,:])
                    @views temp -= BLAS.dot(tiJaB_i[i,:,a,b],Fmi[:,j])#sum( tiJaB_i[i,:,a,b] .* Fmi[:,j])
                    @views temp -= BLAS.dot(tiJaB_i[:,j,a,b], Fmi[:,i])#sum( tiJaB_i[:,j,a,b] .* Fmi[:,i])
                    for m in rocc
					    #term 6
					    for n in rocc
					    	temp += tiJaB_i[m,n,a,b]*Wmnij[m,n,i,j]
					    end
                    end
                    tiJaB_d[i,j,a,b] = temp
                    tiJaB_d[j,i,b,a] = temp
				end
			end
		end
	end
	tiJaB_d .= tiJaB_d ./ Dijab
	return tiJaB_d
end

function form_Wmnij(iJaB,tiJaB)
	nocc = size(tiJaB,1)
	nvir = size(tiJaB,4)
	Wmnij = zeros(Float64,nocc,nocc,nocc,nocc)
    form_Wmnij!(Wmnij,iJaB,tiJaB)
    return Wmnij
end
@fastmath @inbounds function form_Wmnij!(Wmnij,iJaB::Array{Float64,4},tiJaB::Array{Float64,4})
	#Wmnij for RCCD
	#fills Wmnij (!)	
	nocc = size(tiJaB,1)
	nvir = size(tiJaB,4)
	@views iJaB_oovv = iJaB[1:nocc,1:nocc,nocc+1:nocc+nvir,nocc+1:nocc+nvir]
	rocc = collect(UnitRange(1,nocc))
	rvir = collect(UnitRange(1,nvir))

	@views Wmnij .= iJaB[1:nocc,1:nocc,1:nocc,1:nocc]
	for f in rvir
	    for e in rvir
			for j in rocc
                for n in rocc
			        for i in rocc
	                    for m in rocc
							Wmnij[m,n,i,j] += tiJaB[i,j,e,f]*iJaB_oovv[m,n,e,f]/2.0
						end
					end
				end
			end
		end
	end
	return Wmnij
end

function form_Wabef(iJaB,tiJaB)
	nvir = size(tiJaB,4)
	Wabef = zeros(Float64,nvir,nvir,nvir,nvir)
    form_Wabef!(Wabef,iJaB,tiJaB)
    return Wabef
end
@fastmath @inbounds function form_Wabef!(Wabef,iJaB::Array{Float64,4},tiJaB::Array{Float64})
	nocc = size(tiJaB,1)
	nvir = size(tiJaB,4)
    #Wabef .= 0
	Wabef .= iJaB[nocc+1:nvir+nocc,nocc+1:nvir+nocc,nocc+1:nvir+nocc,nocc+1:nvir+nocc]
	rocc = collect(UnitRange(1,nocc))
	rvir = collect(UnitRange(1,nvir))
	@views iJaB_oovv = iJaB[1:nocc,1:nocc,nocc+1:nocc+nvir,nocc+1:nocc+nvir]
    _iJaB_oovv = zeros(Float64,nocc,nocc)
    for f in rvir
        for e in rvir
            _iJaB_oovv .= iJaB_oovv[:,:,e,f]./2.0
	        for b in rvir
	            for a in rvir
					for n in rocc
                        #@views Wabef[a,b,e,f] += BLAS.dot(tiJaB[:,n,a,b],_iJaB_oovv[:,n])
						for m in rocc
                            Wabef[a,b,e,f] += tiJaB[m,n,a,b]*_iJaB_oovv[m,n]#*iJaB_oovv[m,n,e,f]
						end
					end
				end
			end
		end
	end

	return Wabef
end

function form_WmBeJ(iJaB,tiJaB)
	nocc = size(tiJaB,1)
	nvir = size(tiJaB,4)
	WmBeJ = zeros(Float64,nocc,nvir,nvir,nocc)
    form_WmBeJ!(WmBeJ,iJaB,tiJaB)
    return WmBeJ
end
@fastmath @inbounds function form_WmBeJ!(WmBeJ, iJaB, tiJaB)
	nocc = size(tiJaB,1)
	nvir = size(tiJaB,4)
    #WmBeJ .= 0 
	@views WmBeJ .= iJaB[1:nocc,nocc+1:nocc+nvir,nocc+1:nocc+nvir,1:nocc]
	rocc = collect(UnitRange(1,nocc))
	rvir = collect(UnitRange(1,nvir))
	@views iJaB_oovv = iJaB[1:nocc,1:nocc,nocc+1:nocc+nvir,nocc+1:nocc+nvir]
    _iJaB_oovv = zeros(Float64,nocc,nocc)
    for e in rvir
	    for f in rvir
            _iJaB_oovv .= iJaB_oovv[:,:,e,f]./2.0
	        for m in rocc
	            for b in rvir
				    for j in rocc
						for n in rocc
                            WmBeJ[m,b,e,j] -= tiJaB[j,n,f,b]*_iJaB_oovv[m,n]
							WmBeJ[m,b,e,j] += tiJaB[n,j,f,b]*_iJaB_oovv[m,n]*2.0
							WmBeJ[m,b,e,j] -= tiJaB[n,j,f,b]*_iJaB_oovv[n,m]
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
    form_WmBEj!(WmBEj,iJaB,tiJaB)
    return WmBEj
end
@fastmath @inbounds function form_WmBEj!(WmBEj,iJaB,tiJaB)
	nocc = size(tiJaB,1)
	nvir = size(tiJaB,4)
	rocc = collect(UnitRange(1,nocc))
	rvir = collect(UnitRange(1,nvir))
	iJaB_2134 = permutedims(iJaB,[2,1,3,4])
	@views iJaB_ovvo = iJaB_2134[1:nocc,nocc+1:nocc+nvir,nocc+1:nocc+nvir,1:nocc]
	@views iJaB_oovv = iJaB[1:nocc,1:nocc,nocc+1:nocc+nvir,nocc+1:nocc+nvir]
	WmBEj .= -1.0 .* iJaB_ovvo
    _iJaB_oovv = zeros(Float64,nocc,nocc)
	for e in rvir
		for f in rvir
            _iJaB_oovv .= iJaB_oovv[:,:,e,f]./2.0
        	for m in rocc
	            for b in rvir
				    for j in rocc
					    for n in rocc
                            WmBEj[m,b,e,j] += tiJaB[j,n,f,b]*_iJaB_oovv[n,m]
						end
					end
				end
			end
		end
	end
	return WmBEj
end
end #module CC
