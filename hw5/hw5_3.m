function [ nu_list, pi_list ] = hw5_3()
    % Weight Decay in Neural Network

    train = load( 'hw5_3_train.dat' );
    test = load( 'hw5_3_test.dat' );
    
    xdim = 2;                           % dimensions of data
    M = 8;                              % # of perceptron in layer "1"
    K = 0.02;                           % random range
    ETA = 0.05;                         % learning rate
    T = 5000;                           % update itertation
    LAMBDA = [0, 0.001, 0.01, 0.1, 1];  % weight decay
    
    count = 0;    
    for m = M
        for k = K
            for eta = ETA
                for lambda = LAMBDA
                    W1 = mod( rand(xdim+1, m), k ) - k/2; % located on [-k, k]
                    W2 = mod( rand(m+1, 1), k ) - k/2;

                    nu_list = [];
                    pi_list = [];
                    for t = 1:T                        
                        r = randperm( length(train) );
                        x = train( r(1), 1:xdim );
                        x = x';
                        y = train( r(1), xdim+1 );
                        [X0, X1, X2] = forward( W1, W2, m, x, xdim );
                        
                        [delta1, delta2] = backprop( W2, X1, X2, y );
                        
                        delta1 = delta1( 1:m, 1);
                        W1 = ( 1 - eta*lambda )*W1 - eta.*E( X0, delta1 );
                        W2 = ( 1 - eta*lambda )*W2 - eta.*E( X1, delta2 );   

                        nu = error_count( W1, W2, m, train, xdim );
                        pi = error_count( W1, W2, m, test, xdim );
                        nu_list = [nu_list, nu];
                        pi_list = [pi_list, pi];            

                    end

                    count = count + 1;               
                    show( nu_list, pi_list, lambda, count, m, k, eta, T );
                end 
            end
        end        
    end
    
    
    
return;

function ret = E( X, W )
    ret = X*W';
return;

function [X0, X1, X2] = forward( W1, W2, m, x, xdim )
    X0 = [x; -1];
    
    X1 = [];
    for n = 1:m
        x1 = 0;
        for k = 1:xdim+1
            x1 = x1 + ( W1(k,n) .* X0(k) ); 
        end
        x1 = tanh( x1 );
        X1 = [X1; x1];
    end
    X1 = [X1; -1];
    
    X2 = ( sum( X1.*W2 ) ); % linear output -> delta2 should modify
return;

function [delta1, delta2] = backprop( W2, X1, X2, y )
    %delta2 = 2 * (X2 - y) * ( 1 - X2^2 );
    delta2 = 2 * (X2 - y);
    delta1 = delta2 .* W2 .* ( 1 - X1.^2 );
return;

function error = error_count( W1, W2, m, data, xdim )
    [row, col] = size( data );
    error = 0;
    for n = 1:row
        x = data(n, 1:xdim);
        [X0, X1, X2] = forward( W1, W2, m, x', xdim );
        
        y = data(n, xdim+1);
        if ( sign( X2 ) ~= sign( y ) )
            error = error + 1;
        end
    end
    error = error / row;
return;


function show( nu_list, pi_list, lambda, count, m, k, eta, T )
    t = 1:T;
    plot( t, nu_list, t, pi_list);
    titlename = sprintf( 'The NN Training Result of (M, range, \\eta, \\lambda ) = (%d, %0.2f, %0.2f, %0.3f)', m, k, eta, lambda );
    title(titlename);
    xlabel( 't-th iteration' );
    ylabel( 'Error rate' );
    legend( '\nu', '\pi');
    outputpath = sprintf( 'outputs/5.3/%d.png', count );
    saveas( gcf, outputpath, 'png' );
return;
