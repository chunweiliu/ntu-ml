function fx = kerrbf( x1, x2, sigma )
%KERRBF  Gaussian-RBF kernel
%   INPUTS:     x1 -- training pattern           
%               x2 -- training pattern           
%               sigma -- kernel degree
%   OUTPUTS:    fx -- kernel function value

fx = exp( -(norm(x1-x2) )^2 / sigma^2);
