function  [nu_list, pi_list] = rbfn( trainfile, testfile )

    train = load( trainfile );
    test = load( testfile );
    [row, col] = size( train );
    xdim = col - 1;                     % dimensions of data
    
    M_list = [3, 6, 9];                 % how many mountains
    sigma_list = [0.1, 0.5, 2.5];       % the learning rate

    nu_list = [];
    pi_list = [];
    filenum = 1;
    for M = M_list
        % k-means find nu

        X = train( :, 1:xdim );

        nu_list_r = [];
        pi_list_r = [];
        for sigma = sigma_list
            
            mu_list = K_means( M, X );
           
            [alpha_list, mu_list] = radial_basis_function_network( M, sigma, trainfile, mu_list );
                        
            nu = error_count( alpha_list, mu_list, sigma, train );
            pi = error_count( alpha_list, mu_list, sigma, test );
            
            nu_list_r = [ nu_list_r, nu ];
            pi_list_r = [ pi_list_r, pi ];
            
            show(alpha_list, mu_list, train, filenum, sigma, M);
            filenum = filenum + 1;
        end
        
        nu_list = [ nu_list; nu_list_r ];
        pi_list = [ pi_list; pi_list_r ];
    end
    
    
return;

function ret = error_count( alpha_list, mu_list, sigma, data )
    xdim = 2;
    X = data(:, 1:xdim);
    Y = data(:, 1+xdim);
    row = length( Y );
    
    error = 0;
    for n = 1:row
        x = X(n, :);
        
        M = length( alpha_list );
        g = 0;
        for m = 1:M
            alpha = alpha_list( m );
            mu = mu_list( m, : );
            g = g + alpha * gaussian( mu, sigma, x );
        end
        
        g = sign( g );
        if g ~= sign( Y( n ) )
            error = error + 1;
        end
    end
    ret = error / row;
return;

function ret = gaussian( mu, sigma, x )
    ret = exp( (-norm( x - mu )^2) / (2*(sigma^2)) );
return;

function show( alpha_list, mu_list, data, filenum, sigma, M )
    format long;
    plot( mu_list(:, 1), mu_list(:, 2), '*k');
    
    hold on;
    % boundary

    c = [0.836; 0.66; 0.21];
    for a = -1:0.001:1
        
        for b = -1:0.001:1
            x = [a, b];
            M = length( alpha_list );
            
            g = 0;
            for m = 1:M
                alpha = alpha_list( m );
                mu = mu_list( m, : );
                g = g + alpha * gaussian( mu, sigma, x );
            end
            
            signnow = sign( g );
            if ( b ~= -1 && signnow ~= signpre )
                plot( a, b, '.', 'color', c );
            end
            signpre = signnow;
            
        end
    end
    
    % data
    [row, col] = size( data );
    for n = 1:row
        if ( data(n, 3) == 1 )
            plot( data(n, 1), data(n, 2), 'or' );
        else
            plot( data(n, 1), data(n, 2), 'xb' );
        end
    end
    hold off;
    titlename = sprintf( 'The RBFN Training Result of (\\sigma, M) = (%0.1f, %d)', sigma, M );
    title(titlename);
    xlabel( 'x-corrdinate' );
    ylabel( 'y-corrdinate' );
    legend( '\mu_m', 'boundary' );
    mkdir( 'outputs/RBFN' );
    outputpath = sprintf( 'outputs/RBFN/%d.png', filenum );
    saveas( gcf, outputpath, 'png' );
return;

function center = K_means(K, data)
trainNum = size(data,1);

%init k centers
for k = 1 : K
    center(k,:) = data(k,1:2);
end

run_flag = true;
while(run_flag)
    %cluster data
    cluster = [];
    cluster_count(1:K) = 0;
    for i = 1 : trainNum
        minD = 10000000;
        for k = 1 : K
            diff = data(i,:) - center(k,:);
            diff = diff.^2;
            dist = sum(diff);

            if( dist < minD )
                idx = k;
                minD = dist;
            end
        end    
        cluster = [cluster idx];
        cluster_count(idx) = cluster_count(idx) + 1;
    end

    %get new centers
    newcenter(1:K,1:2) = 0;
    for i = 1 : trainNum
        idx = cluster(i);
        newcenter(idx,:) = newcenter(idx,:) + data(i,:);
    end
    for k = 1 : K
        newcenter(k,:) = newcenter(k,:)/cluster_count(k);
    end
    
%    disp(center);
%    disp('-----');
%    disp(newcenter);
 
    %check stop criteria
    run_flag = false;
    for k = 1 : K
        if( sum((newcenter(k,:)-center(k,:)).^2) > 0.0001 )
            run_flag = true;
            break;
        end
    end    
    center = newcenter;
end

