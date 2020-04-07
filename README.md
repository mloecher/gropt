<p align="left">
  <a href="https://github.com/mloecher/gropt/">
    <img src="docs/gropt_logo.png" height="110">
  </a>
</p>

A toolbox for MRI Gradient Optimization (GrOpt)

## Features
* Fast numerical optimizations for MR gradient waveform design (typically 1-100ms).
* Core libraries built completely in C for native integration in pulse sequences.
* Python and Matlab wrappers for easy prototyping.
* Flexible constraint system to enable a range of application.
* Constraints applied in a modular fashion, so adding additional ones is relatively straightforward.

## Contents
- [Features](#features)
- [Installation](#installation)
- [Demos](#demos)
- [Documentation](#documentation)

## Updates

#### ----
 * OpenMP version of TE finder is better implemented. See demo in the `test_TE_finder` function of src/optimize_kernel.c, which will call multiple GrOpt evals simultaneously to find the shortest feasible TE.
#### ----
 * AR-SDMM solver is in its own branch (arsdmm), currently merging 
#### ----
 * Added minTE_finder in src/optimize_kernel.c (minTE_diff function) to more efficiently fine the minimum TE
 * Added simultaneuous axis optimization, contorlled with `Naxis` argument to optimize calls


## Installation

The optimization is written in C and can be found in the src/ directory.

A very basic idea of how to compile for C is included in src/make.txt, however modifications would need to made for input and output as it only runs a test case when run in C.  For easier usage use one of the wrappers:

### Python

The Python module has been tested primarily with [Anaconda](https://www.anaconda.com/) and Python 3.7, though it should work with any type of Python environment.

The setup.py file will build the python module.  To build you can run 
```bash
python setup.py build_ext --inplace
```
from within the python/ directory.  

This will use Cython to generate the source files for a Python module, and then compile it within the GrOpt folder.  For MacOS this procedure requires Xcode.  For Windows you may need a Visual Studio compiler (the free 2019 community version works just fine).  Some common binaries are included in the repository, which should work without any compilation for most.

### Matlab

Assuming you have mex setup correctly (check with `mex -setup`), the two main functions can be compiled by simply running the `make.m` script. 

## Demos

Example usage cases are provided for the Python and Matlab wrappers.  Examples for C applications are shown at the bottom of `src/optimize_kernel.c`

### Python

Demos for Python are all in the form of Jupyter notebooks (.ipynb files) in the ./python/ folder.  Running `jupyter notebook` in the folder will get you started.  Examples show diffusion and non-diffusion gradient design, and most combinations of constraints.

### Matlab

Demo Matlab scripts start with demo_*, are in the ./matlab/ folder, and can be run as is to see some example usage cases.

## Documentation

Further documentation, including descriptions of all constraints and their arguments and units, can be found at http://gropt.readthedocs.io
