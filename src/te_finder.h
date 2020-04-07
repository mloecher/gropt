void minTE_diff(double **G_out, int *N_out, double **ddebug, int verbose,
                double dt0, double gmax, double smax, double search_bval,
                int N_moments, double *moments_params, double PNS_thresh, 
                double T_readout, double T_90, double T_180, int diffmode, double dt_out,
                int N_eddy, double *eddy_params, double slew_reg);

void minTE_diff_par(double **G_out, int *N_out, double **ddebug, int verbose,
                double dt0, double gmax, double smax, double search_bval,
                int N_moments, double *moments_params, double PNS_thresh, 
                double T_readout, double T_90, double T_180, int diffmode, double dt_out,
                int N_eddy, double *eddy_params, double slew_reg);

void minTE_diff_par_worker(double *res, int N_delim, int N, double T_lo, double T_hi, int use_dt, 
                            double **G_out, int *N_out, int verbose,
                            double dt0, double gmax, double smax, double search_bval,
                            int N_moments, double *moments_params, double PNS_thresh, 
                            double T_readout, double T_90, double T_180, int diffmode, double dt_out,
                            int N_eddy, double *eddy_params, double slew_reg);