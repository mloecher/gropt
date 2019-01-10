function res = get_min_TE( target_bval, min_TE, max_TE, gmax, smax, MMT, T_readout, T_90, T_180, dt, diffmode )
%GET_MIN_TE Summary of this function goes here
%   Detailed explanation goes here

a = min_TE;
b = max_TE;
h = b - a;
tol = dt*1e3;

invphi = (sqrt(5) - 1) / 2;                                                                                                      
invphi2 = (3 - sqrt(5)) / 2;

n = ceil(log(tol/h)/log(invphi));

c = a + invphi2 * h;
d = a + invphi * h;

% disp([a,b,c,d]);

Gc = mex_CVXG_fixdt(gmax, smax, MMT, c, T_readout, T_90, T_180, dt, diffmode);
yc = abs(get_bval(Gc, T_readout, dt) - target_bval);

Gd = mex_CVXG_fixdt(gmax, smax, MMT, d, T_readout, T_90, T_180, dt, diffmode);
yd = abs(get_bval(Gd, T_readout, dt) - target_bval);

fprintf('Finding TE (%d iterations): ', n);
for k = 1:n
    fprintf('%d ', k);
    if (yc < yd)
        b = d;
        d = c;
        yd = yc;
        h = invphi*h;
        c = a + invphi2 * h;
        Gc = mex_CVXG_fixdt(gmax, smax, MMT, c, T_readout, T_90, T_180, dt, diffmode);
        yc = abs(get_bval(Gc, T_readout, dt) - target_bval);
    else
        a = c;
        c = d;
        yc = yd;
        h = invphi*h;
        d = a + invphi * h;
        Gd = mex_CVXG_fixdt(gmax, smax, MMT, d, T_readout, T_90, T_180, dt, diffmode);
        yd = abs(get_bval(Gd, T_readout, dt) - target_bval);
    end
end

% bvalc = get_bval(Gc, T_readout, dt);
% bvald = get_bval(Gd, T_readout, dt);
% if (bvalc < bvald)
%     res = Gd;
% else
%     res = Gc;
% end

if (yc < yd)
    res = Gc;
else
    res = Gd;
end

bval_out = get_bval(res, T_readout, dt);
TE_end = numel(res)*dt*1e3+T_readout;
if ( get_bval(res, T_readout, dt) < 0.90 * target_bval )
    fprintf(' TE = %f  bval = %f not big enough, restarting search\n', TE_end, bval_out );
    res = get_min_TE( target_bval, 0.9*TE_end, 2*TE_end, gmax, smax, m0_tol, m1_tol, m2_tol, T_readout, T_90, T_180, dt, diffmode );
end

fprintf(' Done  TE = %f\n', numel(res)*dt*1e3+T_readout);

end

