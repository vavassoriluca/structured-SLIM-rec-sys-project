from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy


ext_modules = [Extension("SLIM_Elastic_Net_Cython",["SLIM_Elastic_Net/Cython/SLIM_Elastic_Net_Cython.pyx"])]

setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = ext_modules,
    include_dirs = [numpy.get_include()]
)