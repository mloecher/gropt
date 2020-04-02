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

# This is all the cpp files that need compiling
sources = ['cg_iter', 'op_bval', 'op_gradient', 'op_main', 'op_moments', 'op_slew', 'gropt_params', 'optimize']
sourcefiles = ['gropt.pyx',] + ['../src/%s.cpp' % x for x in sources]

include_dirs = [".",  "../src", numpy.get_include()]
library_dirs = [".", "../src"]

# openmp stufff here
if is_platform_windows:
    extra_compile_args = []
else:
    extra_compile_args = []


extensions = [Extension("gropt",
                sourcefiles,
                language = "c++",
                include_dirs = include_dirs,
                library_dirs = library_dirs,
                extra_compile_args = extra_compile_args,
            )]

setup(
    ext_modules = cythonize(extensions,
                  compiler_directives={'language_level' : sys.version_info[0]})
)