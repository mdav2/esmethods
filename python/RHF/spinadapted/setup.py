from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
    include_dirs=[numpy.get_include()],
    ext_modules=cythonize("ccsd.pyx",include_path=["/home/mmd01986/anaconda3/envs/psi4/include/python3.7m",numpy.get_include()])
)
