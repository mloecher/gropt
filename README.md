# GrOpt Toolbox
A toolbox for MRI gradient optimization (GrOpt)
# About
The toolbox allows for gradient waveforms to be designed that better utilize available gradient hardware performance, while allowing for additional constraints to be flexibly added to control for a range of desired performances. Additionally, the optimization has been fine-tuned to operate in real-time, allowing for flexible implementation on vendor-agnostic scanner hardware for on-the-fly usage.
# Installation
The optimization is written in C and can be found in the src/ directory.

A very basic idea of how to compile for C is included in src/make.txt, however modifications would need to made for input and output as it only runs a test case when run in C.  For easier usage use one of the wrappers:

### Python

The setup.py file will build the python module.  To build you can run `python setup.py build_ext --inplace`.  

For MacOS you may need to change the compiler being used, in the first few lines of setup.py.  It is hard coded to use a homebrew install of gcc, however any compiler command should work here.

### Matlab

See make.m for an example compile

# More
An example use case is shown in python/demo_v1.ipynb which github will render for you in the browser if you click on it.
