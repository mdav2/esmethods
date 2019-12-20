using HDF5

fid=h5open("/tmp/test2.h5","r")
g=fid["mygroup"]
dset=g["A"]
println(dset[2:3,1:3])
close(fid)
