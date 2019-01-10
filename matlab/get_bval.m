function bval = get_bval(G, T_READOUT, dt)
%GET_BVAL Summary of this function goes here
%   Detailed explanation goes here
    TE = numel(G)*dt*1e3+T_READOUT;

    tINV = floor(TE/dt/1e3/2);

    GAMMA   = 42.58e3; 
    INV = ones(numel(G),1);   
    INV(tINV:end) = -1;

    Gt = cumsum(G'.*INV.*dt);
    bval = sum(Gt.*Gt);
    bval = bval * (GAMMA*2*pi)^2 * dt;
end

