from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=[
        Extension("gower_matrix_c", ["metrics_modules/distance_metrics/gower_matrix_c.pyx"],
                  include_dirs=[numpy.get_include()]),
    ],
)

setup(
    name='gower_matrix_c',
    ext_modules=cythonize("metrics_modules/distance_metrics/gower_matrix_c.pyx"),
)