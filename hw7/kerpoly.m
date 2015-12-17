function fx = kerpoly( x1, x2, d )
%POLYKER Polynomial kernel
%   INPUTS:     x1 -- training pattern           
%               x2 -- training pattern           
%               d -- kernel degree
%   OUTPUTS:    fx -- kernel function value

fx = (1 + dot(x1,x2, 2)) .^ d;
%fx = (1 + x1*x2') .^ d;