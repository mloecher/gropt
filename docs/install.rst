Installation
=================================

GrOpt is written in C, with many wrapper and helper functions written in Matlab and Python.

C Compilation
==============

The main optimization routines are written and can be directly compiled.  Example usages are shown at the bottom of ``src/optimize_kernel.c``.  All of the functions that are would be called by the user are found in ``src/optimize_kernel.c``

Python
==============

The optimization routines have been wrapped with Cython, they can be built simply running in the python/ directory: ::

    python setup.py build_ext --inplace

Which will build the python module named ``gropt`` in the python directory, which can then be imported into any python code.

Example jupyter notebooks are provided in the python/ directory that show many ways how the module can be used.

Matlab
===============

There is set of mex functions to use for calling the optimization routines from Matlab.  They can be compiled by running the file ``make.m`` from within the matlab directory.

The matlab/ directory then has sample scripts that show the verious ways to call the optimization functionss.