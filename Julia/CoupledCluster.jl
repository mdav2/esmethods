module CoupledCluster
"""
Basic module for running CC computations in Julia.

short term goal is RCCD and UCCD
medium term goal is RCCSD and UCCSD
long term goal is RCCSD(T) and UCCSD(T)

Implemented --> RCCD
			--> 
Optimized --> RCCD


usage --> methods should be defined like do_<r/u><method> and take in
	  --> a Wavefunction.jl Wfn object as their sole _required_ input.
	  --> optional inputs such as maxit, convergence, etc can be defined
	  --> via multiple dispatch
"""

using Wavefunction
using LinearAlgebra
using Dates
export do_rccd


function do_rccd(refWfn::Wfn)
    #Implicit dtype = float64
	#implicit maxit = 40
    return do_rccd(refWfn,40)
end
@fastmath @inbounds function do_rccd(refWfn::Wfn,maxit)
    #goes through appropriate steps to do RCCD
    nocc = refWfn.nalpha
    nvir = refWfn.nvira
    iJaB = permutedims(refWfn.pqrs,[1,3,2,4])
	dtt = eltype(iJaB)
    epsa = refWfn.epsa
	T2 = zeros(dtt,nocc,nocc,nvir,nvir)
    Dijab = form_Dijab(T2,epsa)
    T2_init!(T2,iJaB,Dijab)
    println(ccenergy(T2,iJaB))
    Fae = form_Fae(T2,iJaB)
    Fmi = form_Fmi(T2,iJaB)
	println("Formed 1particles")
    Wmnij = form_Wmnij(iJaB,T2)
	println("Formed 1particles")
	Wabef = form_Wabef(iJaB,T2)
	println("Formed 1particles")
	WmBeJ = form_WmBeJ(iJaB,T2)
	println("Formed 1particles")
	WmBEj = form_WmBEj(iJaB,T2)
	println("Formed 1particles")
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

@fastmath @inbounds function cciter(tiJaB_i,iJaB,Dijab,Fae,Fmi,Wabef,Wmnij,WmBeJ,WmBEj)
    form_Fae!(Fae,tiJaB_i,iJaB)
    form_Fmi!(Fmi,tiJaB_i,iJaB)
    form_Wmnij!(Wmnij,iJaB,tiJaB_i)
	form_Wabef!(Wabef,iJaB,tiJaB_i)
	form_WmBeJ!(WmBeJ,iJaB,tiJaB_i)
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
	dt = eltype(iJaB)
	nvir = size(tiJaB,4)
    Fae = zeros(dt,nvir,nvir)
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
    for f in rvir
    	for a in rvir
        	for e in rvir
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
	dt = eltype(iJaB)
	nocc = size(tiJaB,1)
	nvir = size(tiJaB,4)
    Fmi = zeros(dt,nocc,nocc)
    form_Fmi!(Fmi,tiJaB,iJaB)
	return Fmi
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
                    @simd for m in rocc
                        Fmi[m,i] += tiJaB[i,n,e,f]*(2*iJaB_oovv[m,n,e,f] - iJaB_oovv[m,n,f,e])
                    end
                end
            end
        end
    end
    #return Fmi
end

function form_Dijab(tiJaB,F)
	dt = eltype(tiJaB)
	nocc = size(tiJaB,1)
	nvir = size(tiJaB,4)
	rocc = collect(UnitRange(1,nocc))
	rvir = collect(UnitRange(1,nvir))
	Dijab = zeros(dt,nocc,nocc,nvir,nvir)
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

function form_T2(tiJaB_i,Fae,Fmi,WmBeJ,WmBEj,Wabef,Wmnij,iJaB,Dijab)
	dtt = eltype(tiJaB_i)
	nocc = size(Wmnij,1)
	nvir = size(tiJaB_i,4)
	tiJaB_d = zeros(dtt,nocc,nocc,nvir,nvir)
	@views iJaB_oovv = iJaB[1:nocc,1:nocc,nocc+1:nocc+nvir,nocc+1:nocc+nvir]
	rocc = collect(UnitRange(1,nocc))
	rvir = collect(UnitRange(1,nvir))
    _Wabef = zeros(dtt,nvir,nvir)
	ttiJaB_i = permutedims(tiJaB_i,[4,1,2,3])
	WWmBEj  = permutedims(WmBEj,[3,1,2,4])
	WWmBeJ  = permutedims(WmBeJ,[3,1,2,4])
	WmBEj = nothing
	WmBeJ = nothing
	for b in rvir
        for a in rvir #collect(UnitRange(b+1,nvir))
            _Wabef .= Wabef[a,b,:,:]
		    for j in rocc
                for i in rocc#collect(UnitRange(j+1,nocc))
                    #term 2
                    temp = iJaB_oovv[i,j,a,b]
                    #@views T2_ija = tiJaB_i[i,j,a,:]
                    @views temp += BLAS.dot(tiJaB_i[i,j,a,:] , Fae[b,:])#reduce(+, tiJaB_i[i,j,a,:] .* Fae[b,:] )
                    @views temp += BLAS.dot(tiJaB_i[i,j,:,b], Fae[a,:])#reduce(+, tiJaB_i[i,j,:,b] .* Fae[a,:] )

					@simd for m in rocc
                        #@views temp += BLAS.dot(ttiJaB_i[:,m,i,b],WWmBEj[:,m,a,j])
                        #@views temp += BLAS.dot(ttiJaB_i[:,m,j,a],WWmBEj[:,m,b,i])
						#@views temp += BLAS.dot(ttiJaB_i[:,i,m,a],WWmBeJ[:,m,b,j])
						#@views temp -= BLAS.dot(ttiJaB_i[:,m,i,a],WWmBeJ[:,m,b,j])
						#@views temp += BLAS.dot(ttiJaB_i[:,i,m,a],WWmBeJ[:,m,b,j])
						#@views temp += BLAS.dot(ttiJaB_i[:,i,m,a],WWmBEj[:,m,b,j])
						#@views temp += BLAS.dot(ttiJaB_i[:,j,m,b],WWmBeJ[:,m,a,i])
						#@views temp -= BLAS.dot(ttiJaB_i[:,m,j,b],WWmBeJ[:,m,a,i])
						#@views temp += BLAS.dot(ttiJaB_i[:,j,m,b],WWmBeJ[:,m,a,i])
						#@views temp += BLAS.dot(ttiJaB_i[:,j,m,b],WWmBEj[:,m,a,i])
						
                        @views temp += dot(ttiJaB_i[:,m,i,b],WWmBEj[:,m,a,j])
                        @views temp += dot(ttiJaB_i[:,m,j,a],WWmBEj[:,m,b,i])
						@views temp += dot(ttiJaB_i[:,i,m,a],WWmBeJ[:,m,b,j])
						@views temp -= dot(ttiJaB_i[:,m,i,a],WWmBeJ[:,m,b,j])
						@views temp += dot(ttiJaB_i[:,i,m,a],WWmBeJ[:,m,b,j])
						@views temp += dot(ttiJaB_i[:,i,m,a],WWmBEj[:,m,b,j])
						@views temp += dot(ttiJaB_i[:,j,m,b],WWmBeJ[:,m,a,i])
						@views temp -= dot(ttiJaB_i[:,m,j,b],WWmBeJ[:,m,a,i])
						@views temp += dot(ttiJaB_i[:,j,m,b],WWmBeJ[:,m,a,i])
						@views temp += dot(ttiJaB_i[:,j,m,b],WWmBEj[:,m,a,i])
                    #    for e in rvir
							#8
						#	temp += tiJaB_i[i,m,a,e]*WmBeJ[m,b,e,j]
						#	temp -= tiJaB_i[m,i,a,e]*WmBeJ[m,b,e,j]
							#9
					#		temp += tiJaB_i[i,m,a,e]*(WmBeJ[m,b,e,j])
					#		temp += tiJaB_i[i,m,a,e]*(WmBEj[m,b,e,j])
							#10
                    #        cache = tiJaB_i[j,m,b,e]
							##13
				#			temp += cache*WmBeJ[m,a,e,i]
				#			temp -= tiJaB_i[m,j,b,e]*WmBeJ[m,a,e,i]
							#13
                      #      temp += cache*(WmBeJ[m,a,e,i])
                      #      temp += cache*(WmBEj[m,a,e,i])
					#	end
                    end
                    @views temp += reduce(+, tiJaB_i[i,j,:,:] .* _Wabef[:,:])
                    @views temp -= BLAS.dot(tiJaB_i[i,:,a,b],Fmi[:,j])#sum( tiJaB_i[i,:,a,b] .* Fmi[:,j])
                    @views temp -= BLAS.dot(tiJaB_i[:,j,a,b], Fmi[:,i])#sum( tiJaB_i[:,j,a,b] .* Fmi[:,i])
					@views temp += reduce(+, tiJaB_i[:,:,a,b] .* Wmnij[:,:,i,j])
                    #for m in rocc
					#    #term 6
					#    for n in rocc
					#    	temp += tiJaB_i[m,n,a,b]*Wmnij[m,n,i,j]
					#    end
                    #end
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
	dtt = eltype(iJaB)
	nocc = size(tiJaB,1)
	nvir = size(tiJaB,4)
	Wmnij = zeros(dtt,nocc,nocc,nocc,nocc)
    form_Wmnij!(Wmnij,iJaB,tiJaB)
    return Wmnij
end
@fastmath @inbounds function form_Wmnij!(Wmnij,iJaB,tiJaB)
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
	                    @simd for m in rocc
							Wmnij[m,n,i,j] += tiJaB[i,j,e,f]*iJaB_oovv[m,n,e,f]/2.0
						end
					end
				end
			end
		end
	end
end

function form_Wabef(iJaB,tiJaB)
	dt = eltype(iJaB)
	nvir = size(tiJaB,4)
	Wabef = zeros(dt,nvir,nvir,nvir,nvir)
    form_Wabef!(Wabef,iJaB,tiJaB)
    return Wabef
end
function form_Wabef!(Wabef,iJaB,tiJaB)
	dtt = eltype(Wabef)
	nocc = size(tiJaB,1)
	nvir = size(tiJaB,4)
    #Wabef .= 0
	Wabef .= iJaB[nocc+1:nvir+nocc,nocc+1:nvir+nocc,nocc+1:nvir+nocc,nocc+1:nvir+nocc]
	rocc = collect(UnitRange(1,nocc))
	rvir = collect(UnitRange(1,nvir))
	@views iJaB_oovv = iJaB[1:nocc,1:nocc,nocc+1:nocc+nvir,nocc+1:nocc+nvir]
    _iJaB_oovv = zeros(dtt,nocc,nocc)
    for f in rvir
        for e in rvir
            _iJaB_oovv .= iJaB_oovv[:,:,e,f]./2.0
	        for b in rvir
	            for a in rvir
					#@views Wabef[a,b,e,f] += reduce(+, tiJaB[:,:,a,b] .* _iJaB_oovv[:,:])
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
end

function form_WmBeJ(iJaB,tiJaB)
	dtt = eltype(iJaB)
	nocc = size(tiJaB,1)
	nvir = size(tiJaB,4)
	WmBeJ = zeros(dtt,nocc,nvir,nvir,nocc)
    form_WmBeJ!(WmBeJ,iJaB,tiJaB)
    return WmBeJ
end
function form_WmBeJ!(WmBeJ, iJaB, tiJaB)
	dtt = eltype(WmBeJ)
	nocc = size(tiJaB,1)
	nvir = size(tiJaB,4)
    #WmBeJ .= 0 
	@views WmBeJ .= iJaB[1:nocc,nocc+1:nocc+nvir,nocc+1:nocc+nvir,1:nocc]
	rocc = collect(UnitRange(1,nocc))
	rvir = collect(UnitRange(1,nvir))
	@views iJaB_oovv = iJaB[1:nocc,1:nocc,nocc+1:nocc+nvir,nocc+1:nocc+nvir]
    _iJaB_oovv = zeros(dtt,nocc,nocc)
    for e in rvir
	    for f in rvir
            _iJaB_oovv .= iJaB_oovv[:,:,e,f]./2.0
	        for b in rvir
				for j in rocc
	        		for m in rocc
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
end


function form_WmBEj(iJaB,tiJaB)
	dtt = eltype(iJaB)
	nocc = size(tiJaB,1)
	nvir = size(tiJaB,4)
	WmBEj = zeros(dtt,nocc,nvir,nvir,nocc)
    form_WmBEj!(WmBEj,iJaB,tiJaB)
    return WmBEj
end
function form_WmBEj!(WmBEj,iJaB,tiJaB)
	dtt = eltype(WmBEj)
	nocc = size(tiJaB,1)
	nvir = size(tiJaB,4)
	rocc = collect(UnitRange(1,nocc))
	rvir = collect(UnitRange(1,nvir))
	#doing this double permutation scheme reduces floating point performance,
	#but also reduces memory footprint
	iJaB = permutedims(iJaB,[2,1,3,4])
	@views iJaB_ovvo = iJaB[1:nocc,nocc+1:nocc+nvir,nocc+1:nocc+nvir,1:nocc]
	WmBEj -= iJaB_ovvo
	iJaB = permutedims(iJaB,[2,1,3,4])
	@views iJaB_oovv = iJaB[1:nocc,1:nocc,nocc+1:nocc+nvir,nocc+1:nocc+nvir]
    _iJaB_oovv = zeros(dtt,nocc,nocc)
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
