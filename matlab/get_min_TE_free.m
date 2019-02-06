function [G_out, T_out] = get_min_TE_free( params, T_hi )
%GET_MIN_TE Summary of this function goes here
%   Detailed explanation goes here

T_lo = 0.0;
T_range = T_hi-T_lo;

best_time = 999999.9;

while ((T_range*1e-3) > (params.dt/4.0))
    params.TE = T_lo + (T_range)/2.0;
    fprintf('Testing TE = %d\n', params.TE);
    [G, lim_break] = gropt(params);
    if lim_break == 0
        T_hi = params.TE;
        if T_hi < best_time
            G_out = G;
            T_out = T_hi;
            best_time = T_hi;
        end
    else
        T_lo = params.TE;
    end
    T_range = T_hi-T_lo;
end

end

