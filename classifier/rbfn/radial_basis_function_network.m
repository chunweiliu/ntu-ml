function [alpha_list, mu_list] = radial_basis_function_network( M, sigma, trainfile, mu_list )
    format long;
    
    train = load( trainfile );   
            
    xdim = 2;
    X = train(:, 1:xdim);
    Y = train(:, 1+xdim);
    
    % k-means find nu
    %mu_list = kmeans( M, X );
    
    % construct phi matrix
    [row, col] = size( X );
    phi = [];
    for n = 1:row
        phi_r = [];
        for m = 1:M
            phi_r = [phi_r, gaussian( mu_list(m, 1:xdim), sigma, X(n, 1:xdim ) )];
        end
        phi = [phi; phi_r];
    end
    
    % linear regression
    alpha_list = inv(phi' * phi ) * phi' * Y;
return;

function ret = gaussian( mu, sigma, x )
    ret = exp( (-norm( x - mu )^2) / (2*(sigma^2)) );
return;
