function [ nu_list, pi_list ] = hw4_4()
    train = load( 'hw4_train.dat' );
    test = load( 'hw4_test.dat' );
    
    M = [1, 3, 5];          % layer 1
    K = [0.02, 0.2, 2];     % random range
    ETA = [0.01, 0.1, 1];   % learning rate
    
    count = 0;    
    for m = M
        for k = K
            for eta = ETA
                
                xdim = 2;
                W1 = mod( rand(xdim+1, m), k ) - k; % located on [-k, k]
                W2 = mod( rand(m+1, 1), k ) - k;
                T = 50000;
                %T = 1;
                nu_list = [];
                pi_list = [];
                E_list = [];
                for t = 1:T
                    r = randperm( length(train) );
                    x = train( r(1), 1:xdim );
                    x = x';
                    y = train( r(1), xdim+1 );
                    [X0, X1, X2] = forward( W1, W2, m, x );
                    [delta1, delta2] = backprop( W2, X1, X2, y );
                    delta1 = delta1( 1:m, 1);
                    W1 = W1 - eta.*E( X0, delta1 );
                    W2 = W2 - eta.*E( X1, delta2 );   
                    
                    nu = error_count( W1, W2, m, train );
                    pi = error_count( W1, W2, m, test );
                    E = error_e( W1, W2, m, train );
                    nu_list = [nu_list, nu];
                    pi_list = [pi_list, pi];
                    E_list = [E_list, E];
                end

                count = count + 1;               
                show( nu_list, pi_list, E_list, count, m, k, eta );
                
            end
        end        
    end
    
    
    
return;

function ret = E( X, W )
    ret = X*W';
return;

function [X0, X1, X2] = forward( W1, W2, m, x )
    X0 = [x; -1];
    
    X1 = [];
    for n = 1:m
        x1 = 0;
        for k = 1:3
            x1 = x1 + ( W1(k,n) .* X0(k) ); 
        end
        x1 = tanh( x1 );
        X1 = [X1; x1];
    end
    X1 = [X1; -1];
    
    X2 = tanh( sum( X1.*W2 ) );
return;

function [delta1, delta2] = backprop( W2, X1, X2, y )
    delta2 = 2 * (X2 - y) * ( 1 - X2^2 );
    delta1 = delta2 .* W2 .* ( 1 - X1.^2 );
return;

function error = error_count( W1, W2, m, data )
    [row, col] = size( data );
    error = 0;
    for n = 1:row
        xdim = 2;
        x = data(n, 1:xdim);
        [X0, X1, X2] = forward( W1, W2, m, x' );
        
        y = data(n, xdim+1);
        if ( sign( X2 ) ~= sign( y ) )
            error = error + 1;
        end
    end
    error = error / row;
return;

function error = error_e( W1, W2, m, data )
    [row, col] = size( data );
    error = 0;
    for n = 1:row
        xdim = 2;
        x = data(n, 1:xdim);
        [X0, X1, X2] = forward( W1, W2, m, x' );
        
        y = data(n, xdim+1);
        
        error = error + (y - X2)^2;
    end
    error = error / row;
return;

function show( nu_list, pi_list, E_list, count, m, k, eta )
    t = 1:50000;
    plot( t, nu_list, t, pi_list, t, E_list );
    titlename = sprintf( 'The NN Training Result of (M, range, \\eta ) = (%d, %0.2f, %0.2f)', m, k, eta );
    title(titlename);
    xlabel( 't-th iteration' );
    ylabel( 'error' );
    legend( '\nu', '\pi', 'E' );
    %outputpath = '/home/master/97/r97032/htdocs/show/kmeans_result.png';
    outputpath = sprintf( 'outputs/4.4/%d.png', count );
    saveas( gcf, outputpath, 'png' );
return;