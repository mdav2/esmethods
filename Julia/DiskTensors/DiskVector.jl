using HDF5

function todisk(fname::String,A::Array{Float64})

end

struct DiskVector
	file::HDF5File
	group::HDF5Group
	data::HDF5Dataset
	fname::String
	gname::String
	dname::String
	dtype::Union{Type{Float32},Type{Float64}}
	size::Int #number of elements to be allocated
	compress::Bool
	chonk::Bool 
	chonksz::Tuple
end

function DiskVector(fname::String,dtype::Union{Type{Float32},Type{Float64}},
					mode::String="r+",fieldname::String="data",
					size::Int,compress::Bool=false,chonk::Bool=false,chonksz::Array{Int}=Tuple{Int,Int}((1,1)))
	file = h5open(fname,mode)
	if (mode == "w") | (mode == "w+")
		g = g_create(file,"data")
	else
		g = file[fieldname]
	end
	dataset = d_create(g,fieldname,datatype(dtype),dataspace(size),"compress")
	DiskVector(file,fname,dtype,size,compress,chonk,chonksize)
end

function fill!(dvec::DiskVector,arr::Array{Union{Type{Float32},Type{Float64}}})
end

function activate(dvec::DiskVector)
	"opens a DiskVector object for access and obtains a mutex for thread
	safety"
end
function deactivate(dvec::DiskVector)
	"closes a DiskBector object for access and releases mutex"
end
