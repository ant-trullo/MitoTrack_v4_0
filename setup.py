import numpy
from distutils.core import setup
from Cython.Build import cythonize

setup(
      ext_modules=cythonize("SpotsDetectionUtility.pyx", annotate=True)
#      include_dirs=[numpy.get_include()]

      )

# python3 setup.py build_ext --inplace
