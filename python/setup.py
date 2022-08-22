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
sources = ['cg_iter', 'op_bval', 'op_gradient', 'op_main', 'op_moments', 'op_slew', 'op_maxwell',
            'op_duty', 'op_eddy', 'op_eddyspect2', 'op_eddyspect', 'op_btensor', 'op_acoustic', 'op_girfec', 'op_pns',
            'gropt_params', 'optimize', 'diff_utils']
sourcefiles = ['gropt.pyx',] + ['../src/%s.cpp' % x for x in sources]

#include_dirs = [".",  "../src", numpy.get_include()] # MJM
include_dirs=[".",  "./src", "/usr/local/include/", numpy.get_include()],
library_dirs = [".", "../src"]

# openmp stufff here
if is_platform_windows:
    extra_compile_args = []
else:
#    extra_compile_args = []
    extra_compile_args = ["-std=c++11"]  # MJM


# MJM
# extensions = [Extension("gropt",
#                 sourcefiles,
#                 language = "c++",
#                 include_dirs = include_dirs,
#                 library_dirs = library_dirs,
#                 extra_compile_args = extra_compile_args,
#                 # undef_macros=['NDEBUG'], # This will re-enable the Eigen assertions
#             )]

# MJM
extensions = [Extension("gropt",
                    sourcefiles,
                    language="c++",
                    include_dirs=[".",  "./src", "/usr/local/include/", numpy.get_include()],
                    library_dirs=[".", "./src", "/usr/local/lib/"],
                    extra_compile_args=['-std=c++11', "-mmacosx-version-min=10.9"],
                    extra_link_args=["-stdlib=libc++", "-mmacosx-version-min=10.9"],
                   )]

setup(
    ext_modules = cythonize(extensions,
                  compiler_directives={'language_level' : sys.version_info[0]},
                  nthreads=8)
)