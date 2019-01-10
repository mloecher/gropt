function plot_waveform( G, T_readout, dt)
%PLOT_WAVEFORM Summary of this function goes here
%   Detailed explanation goes here

figure();

bval = get_bval(G, T_readout, dt);
TE2 = numel(G)*dt*1e3+T_readout;



subplot(2,1,1);
x = [0:numel(G)-1] .* dt * 1e3;
y = G * 1000;
plot(x,y,'LineWidth', 2);
ylim([-1.1 * max(abs(y)), 1.1 * max(abs(y))])
ylabel('G [mT/m]')
xlabel('t [ms]')
title(['bval = ' num2str(bval) '   TE = ' num2str(TE2)])



tINV = floor(TE2/dt/1e3/2);

INV = ones(numel(G),1);   INV(tINV:end) = -1;
INV = INV';

Nm = 5;
tvec = (0:numel(G)-1)*dt; % in sec
tMat = zeros( Nm, numel(G) );
for mm=1:Nm,
    tMat( mm, : ) = tvec.^(mm-1);
end


moments = dt*tMat.*repmat(G.*INV, size(tMat,1), 1);
moments = cumsum(moments, 2);

subplot(2,1,2);
hold on
for i = 1:3
    plot(x, moments(i,:)/max(abs(moments(i,:))), 'LineWidth', 2);
end

ylabel('Moments [A.U]')
xlabel('t [ms]')
legend('M0', 'M1', 'M2')
hline = refline([0 0]);
hline.Color = 'k';

end

