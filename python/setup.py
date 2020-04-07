# Updated setup.py for MacOS stock LLVM compilation, but I would like to
# add some type of homebrew gcc detection for openmp . . . 
import os, sys

def is_platform_windows():
    return sys.platform == "win32"

def is_platform_mac():
    return sys.platform == "darwin"

try:
    from setuptools import setup
    from setuptools import Extension
except ImportError:
    from distutils.core import setup
    from distutils.extension import Extension

from Cython.Build import cythonize
import numpy


sourcefiles = ['gropt.pyx', '../src/cvx_matrix.c', '../src/te_finder.c', '../src/op_gradient.c', '../src/op_maxwell.c', '../src/op_bval.c', '../src/op_beta.c', '../src/op_eddy.c', '../src/op_slewrate.c', '../src/op_moments.c', '../src/op_pns.c']

include_dirs = [".",  "../src", numpy.get_include()]
library_dirs = [".", "../src"]
if is_platform_windows:
    extra_compile_args = []
else:
    extra_compile_args = ['-std=c11']


extensions = [Extension("gropt",
                sourcefiles,
                language = "c",
                include_dirs = include_dirs,
                library_dirs = library_dirs,
                extra_compile_args = extra_compile_args,
            )]

setup(
    ext_modules = cythonize(extensions)
)