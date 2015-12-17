function [nu_list, pi_list] = naive_bayes( trainfile, testfile )
% naive bayes
% [nu_list, pi_list] = naive_bayes( trainfile, testfile )
    train = load( trainfile );
    test = load( testfile );
    
    xdim = 2;
    
    K = [2, 5, 10, 20, 50];
    
    [prior_pos, prior_neg]= prior( train );
    [train_p, train_n] = split( train, xdim );

    nu_list = [];
    pi_list = [];
    for k = K
        % hist
        bins = bins_generate( k );
        
        % train             
        [rowp, colp] = size( train_p );
        h_p = hist( train_p, bins );
        likelihood_pos = h_p ./ rowp;
        
        [rown, coln] = size( train_n );
        h_n = hist( train_n, bins );
        likelihood_neg = h_n ./ rown;
                
        % test
        nu = error_count( train, prior_pos, prior_neg, likelihood_pos, likelihood_neg, bins, xdim );
        pi = error_count( test, prior_pos, prior_neg, likelihood_pos, likelihood_neg, bins, xdim );
        nu_list = [ nu_list, nu ];
        pi_list = [ pi_list, pi ];
    end
return;

function [prior_pos, prior_neg] = prior( train )
    [row, last] = size( train );

    prior_pos = 0;
    prior_neg = 0;
    for t = 1:row
        y = train( t, last );
        if y == 1
            prior_pos = prior_pos + 1;
        elseif y == -1
            prior_neg = prior_neg + 1;
        end
    end
    prior_pos = prior_pos / row;
    prior_neg = prior_neg / row;
return;

function [train_p, train_n] = split( train, xdim )
    [row, last] = size( train );
    
    train_p = [];
    train_n = [];
    for t = 1:row
        node = train( t, : );
        if ( node( last ) == 1 )
            train_p = [train_p; node(1:xdim)];
        elseif( node( last ) == -1 )
            train_n = [train_n; node(1:xdim)];
        end
    end
return;

function bins = bins_generate( k )
    c = (1/k);
    bin = -c/2;
    
    bins = [];
    for t = 1:k
        bin = bin + c;
        bins = [bins, bin];
    end
return;

function s = g( x, prior_pos, prior_neg, likelihood_pos, likelihood_neg, bins, xdim )
    selmat = select_matrix( x, bins, xdim );
    
    l_pos = sum( selmat .* likelihood_pos );
    likelihood_p = prod( l_pos, 2 );
    
    l_neg = sum( selmat .* likelihood_neg );
    likelihood_n = prod( l_neg, 2 );
    
    term = ( likelihood_p .* prior_pos ) - ...
           ( likelihood_n .* prior_neg );
    
    if term > 0
        s = 1;
    elseif term < 0
        s = -1;
    else
        s = sign ( rand(1,1) - 0.05 );
    end
    
    %term = ( likelihood_p .* prior_pos ) / ...
    %       ( likelihood_n .* prior_neg );
    %s = sign( log( term ) );
return;

function m = select_matrix( x, bins, xdim )
    m = [];
    for t = 1:xdim 
        h = hist( x(t), bins );
        m = [m, h'];
    end
return;

function error = error_count( data, prior_pos, prior_neg, likelihood_pos, likelihood_neg, bins, xdim )
    [row, col] = size( data );
    
    error = 0;
    for n = 1:row
        x = data( n, 1:xdim );
        s = g( x, prior_pos, prior_neg, likelihood_pos, likelihood_neg, bins, xdim );
            
        if s ~= sign( data(n, col) )
            error = error + 1;
        end
    end
    error = error / row;
return;