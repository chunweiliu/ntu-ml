function g = rbfn( x, y, k, sigma )
%RADIAL BASIS FUNCTION NETWORK
% INPUTS:   rbfn( x, y, k, delta ); x is data, y is labels
%                                   k is #means, delta control gaussian
% OUTPUTS:  g(k, ALPHA, MU);        k is #means, ALPHA is weight of gaussian
%                                   MU is the center of gaussian
mu = kmeans( x, k );

% ALPHA
xnum = size(x, 1);
xdim = size(x, 2);
phi = [];
for n = 1:xnum
    phi_row = [];
    for kk = 1:k
        phi_row = [phi_row, gaussian( mu(kk, 1:xdim), sigma, x(n, 1:xdim ) )];
    end
    phi = [phi; phi_row];
end

% Linear Regression
alpha = inv(phi' * phi ) * phi' * y;

g = [];
g{1} = {mu};
g{2} = {alpha};
%g{3} = {gaussian};
return;

function ret = gaussian( mu, sigma, x )
    ret = exp( (-norm( x - mu )^2) / (2*(sigma^2)) );
return;

function centers = kmeans( x, k )
xnum = size(x, 1);
xdim = size(x, 2);

% initial k's centers
r = floor(rand(1, 1) * (xnum - 3) + 2);
r = [r-1, r, r+1];
%r = [1, 2, 3];
for kk = 1:k
    centers(kk, :) = x(r(kk), :);
end

epsilon = 0.0001;
flag = true;
while(flag)
    % cluster
    clusters = [];
    clusters_count(1:k) = 0;
    for n = 1:xnum
        d_min = 1e10;
        for kk = 1:k
            d = norm( x(n,1:xdim ) - centers(kk,1:xdim) );
            
            if( d < d_min )
                idx = kk;
                d_min = d;
            end
        end
        clusters = [ clusters, idx ];
        clusters_count( idx ) = clusters_count( idx ) + 1;
    end
    
    % recompute centers
    centers_new( 1:k, 1:xdim) = 0;
    for n = 1:size(x,1)
        idx = clusters(n);
        centers_new( idx, : ) = centers_new( idx, : ) + x( n, : );
    end
    for kk = 1:k
        centers_new(kk, :) = centers_new(kk, :) ./ clusters_count( kk );
    end

    % check if stop
    
    if abs( sum( sum( centers_new - centers ) ) ) > epsilon;
        flag = true;
    else
        flag = false;
    end
    centers = centers_new;
end
return;