function hx = stumpfunc( h, x )
dir = h(1);
dim = h(2);
theta = h(3);

hx = sign(dir*x(:, dim) - theta);
hx = hx + (hx == 0);
return;

