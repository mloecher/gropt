#############
C Usage
#############

The two primary calls to the C library are the run_kernel_diff() and minTE_diff() functions.

Both take basically the same arguments, except run_kernel_diff() takes a fixed TE, and minTE_diff() takes a b-value to search for.

For usage examples, see the end of src/optimize_kernel.c, basic usages of both functions are shown.

Arguments
=============

Note that this is just a list of all possible arguments, the order might be slightly different for a given function, see definitions in src/optimize_kernel.c

double **G_out
    Pointer to the array where the output will be stored.  This will be allocated within the function.

int *N_out 
    Pointer to an integer specifying the size of G_out after completion of the optimization.

double **ddebug 
    Pointer to the array where debug output information will be stored.  It will have a fixed length 100 (but you should probably check that to see if it has changed)

int verbose
    0 for no message, >0 for debug messages to console

int N
    The size of the output gradient array

double dt
    The raster time of the gradient waveform in seconds, i.e. the distance between timepoints.

double gmax
    The maximum gradient amplitude in T/s.

double smax
    The maximum slew rate in T/m/s.

double TE
    The TE of the waveform in MILLIseconds.  For diffusion waveforms, this takes into account T_readout.  For non-diffusion it is just the length of the waveform, i.e. N*dt.

int N_moments
    The number of moment constraints that will be used.

double *moments_params
    An array describing the moment constraints, with 7 values per constraint as described :ref:`here <ref-moment-constraints>`.  So it is an array of N_moments*7 doubles.

double PNS_thresh    
    PNS threshold with the single exponential model, any decimal should work.  To disable the constraint use -1.0 (or any negative number).

double T_readout, double T_90, double T_180
    Timings of the readout prewinder, initial excitation pulse, and 180 pulse, in MILLIseconds.

double bval_weight, double slew_weight, double moments_weight
    Initial weights for the various constraints, these can be kept at defaults, or all at 1.0, and everything will work well.

double bval_reduce
    How much to change the above weights when convergence has slowed down too much. Dafault to around 10.

double dt_out
    Adds a final linear interpolation to the waveform to reach dt_out in seconds.

int N_eddy
    Number of eddy current constraints, can be 0 to disable

double *eddy_params
    An array describing the eddy current constraints, with 4 values per constraint as described :ref:`here <ref-eddy-constraints>`.  So it is an array of N_moments*4 doubles.

int is_Gin
    1 if there is an input gradient waveform for warm starting the optimization, 0 to disable (default).

double *G_in
    An array holding initialization values for the G vector in the optimization

double search_bval
    When used in run_kernel, this gives a bvalue maximum, so iterations with a higher b-value are immediately stopped.  It can be set to -1 to disable.  

    When referred to in minTE_diff(), it is the bvalue we are searching for.

int N_gfix
    Number of fixed values in G.  0 is off (which technically enforces G=0 at the beginning and end), 2 informs the function that *gfix only has a start and end G value.  N_gfix = N says that *gfix is a full array.

double *gfix
    G values to set as fixed.  Can be 2 entries for start and end.  If full array, then big negative numbers mean NOT fixed, so fill array with -99999 and then fill in fixed values.

double slew_reg
    This describes the slew rate minimization that is applied.  The real amount is actually a multiplier of (1-slew_reg), so 1.0 is nothing, and 0.0 is complete regularization.  Default is 1.0 right now but it might switch to 0.99 for smoother waveforms.
