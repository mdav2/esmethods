struct DiskFourTensor
	fname::String
	dname::String
	sz1::Int
	sz2::Int
	sz3::Int
	sz4::Int
end

function DiskFourTensor(fname::String,dtype::Type,sz1::Int,sz2::Int,sz3::Int,sz4::Int,mode::String="r+")
	file = h5open(fname,mode)
	if (mode == "w") | (mode == "w+")
		dataset = d_create(file,"data",datatype(dtype),dataspace(sz1))
	else
		dataset = file["data"]
	end
	DiskFourTensor(fname,"data",sz1,sz2,sz3,sz4)
end

function blockfill!(dtens::DiskFourTensor,val)
	A = zeros(Float64,dtens.sz1,dtens.sz2,
			  dtens.sz3,dtens.sz4)
	A .= val
	h5write(dtens.fname,"$dtens.dname",A)
end

function d4read(dtens::DiskFourTensor,p1,p2,p3,p4)
	fid = h5open(dtens.fname,"r")
	return fid["$dtens.dname"][p1,p2,p3,p4]
end
function d4write(dtens::DiskFourTensor,p1,p2,p3,p4,val)
	fid = h5open(dtens.fname,"r+")
	fid["$dtens.dname"][p1,p2,p3,p4] = val
end
