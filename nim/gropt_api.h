/*
 * gropt_api.h – C function declarations for the GrOpt Nim bindings.
 *
 * These are the four public entry points exposed by optimize_kernel.c.
 * Nim imports them via {.importc, header: "gropt_api.h".}.
 */
#ifndef GROPT_API_H
#define GROPT_API_H

/* Fixed number of gradient samples (Naxis always 1). */
void run_kernel_diff_fixedN(
    double **G_out, int *N_out, double **ddebug, int verbose,
    int N0, double gmax, double smax, double TE,
    int N_moments, double *moments_params, double PNS_thresh,
    double T_readout, double T_90, double T_180, int diffmode, double dt_out,
    int N_eddy, double *eddy_params, double search_bval, double slew_reg);

/* Fixed number of samples, with an initial gradient waveform (Naxis always 1). */
void run_kernel_diff_fixedN_Gin(
    double **G_out, int *N_out, double **ddebug, int verbose,
    double *G_in, int N0, double gmax, double smax, double TE,
    int N_moments, double *moments_params, double PNS_thresh,
    double T_readout, double T_90, double T_180, int diffmode, double dt_out,
    int N_eddy, double *eddy_params, double search_bval, double slew_reg);

/* Fixed time step, supports Naxis > 1. */
void run_kernel_diff_fixeddt(
    double **G_out, int *N_out, double **ddebug, int verbose,
    double dt0, double gmax, double smax, double TE,
    int N_moments, double *moments_params, double PNS_thresh,
    double T_readout, double T_90, double T_180, int diffmode, double dt_out,
    int N_eddy, double *eddy_params, double search_bval, double slew_reg, int Naxis);

/* Fixed time step with fixed gradient constraints (Naxis always 1). */
void run_kernel_diff_fixeddt_fixG(
    double **G_out, int *N_out, double **ddebug, int verbose,
    double dt0, double gmax, double smax, double TE,
    int N_moments, double *moments_params, double PNS_thresh,
    double T_readout, double T_90, double T_180, int diffmode, double dt_out,
    int N_eddy, double *eddy_params, double search_bval,
    int N_gfix, double *gfix, double slew_reg);

#endif /* GROPT_API_H */
