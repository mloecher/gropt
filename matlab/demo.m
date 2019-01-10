%% Simple solver
gmax = 0.04;
smax = 50.0;
MMT = 2;
TE = 60.0;
T_readout = 12.0;
T_90 = 3.0;
T_180 = 6.0;
dt = 0.10e-3;
diffmode = 2;

G = mex_CVXG_fixdt(gmax, smax, MMT, TE, T_readout, T_90, T_180, dt, diffmode);

plot_waveform(G, T_readout, dt)
%% This will make the exact same waveform as the last one, but with fixed N
gmax = 0.04;
smax = 50.0;
MMT = 2;
TE = 60.0;
T_readout = 12.0;
T_90 = 3.0;
T_180 = 6.0;
N0 = 480;
diffmode = 2;

G = mex_CVXG_fixN(gmax, smax, MMT, TE, T_readout, T_90, T_180, N0, diffmode);

dt = (TE-T_readout) * 1.0e-3 / numel(G);

plot_waveform(G, T_readout, dt)
%% TE finder
target_bval = 100;
min_TE = 60.0; 
max_TE = 100.0;
dt = 0.40e-3;

G = get_min_TE( target_bval, min_TE, max_TE, gmax, smax, MMT, T_readout, T_90, T_180, dt, diffmode );

plot_waveform(G, T_readout, dt)

%%
