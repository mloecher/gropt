GrOpt
=================================

Overview
==========
GrOpt is a toolbox for MRI Gradient Optimization (GrOpT).  Rather than analytical combinations of trapezoids, the software uses optimization methods to generate arbitrary waveforms that are (hopefully) the most time optimal solution.

The main workhorse of the package are the optimization routines written in C, which for final applications are designed to be dropped into a pulse sequence as needed.  Additionally, wrappers for Python and Matlab are provided, as well as some demo cases to show how the software can be used, and to prototype sequences and combinations of constraints.



.. toctree::
    :maxdepth: 2
    :caption: Contents:

    install
    constraints
    usage/index
    citations
    