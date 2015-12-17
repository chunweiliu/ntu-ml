function h = adalearn( x, y, T, learn, func )
[xnum, xdim] = size(x);
u = ones(xnum, 1) ./ xnum;
hw = [];
alp = [];
for t = 1:T
    ht = learn(x, y, u);

    vt = func(ht, x);
    rt = (vt ~= y);
    et = sum(u .* rt) / sum(u);
    
    at = 0.5 * log( (1-et) / (et) );
    
    u = u .* exp( y .* (-at) .* func(ht, x) );
    
    alp = [alp; at];
    hw = [hw; ht];    
end
h = [{T}, {alp}, {hw}];
return;
