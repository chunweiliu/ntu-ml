function ret = kmeans( M, X )

% THE K-MEANS ALGORITHM
% INPUT : X IS A LIST OF ROW VECTORS
%         x1, x2, ... xn ( n-dimension data )
%       : M IS THE NUMBER OF CLUSTERS    
    x = X;
    [row, col] = size( x );

    last = col + 1;

% randomly divide row vectors X to M clusters
    padding = randperm( row );
    %padding = rand( 1, row );
    x( :, last ) = padding';

    clusters_new = x;
    % sort by the same last symbol which mean group 
    [old, idx] = sort( clusters_new(:, last) );
    clusters_new = clusters_new(idx, :);
    
    %[mu_list_new, clusters_new] = mu_m0( clusters_new, M );
    [mu_list_new, clusters_new] = mu_m1( clusters_new, M );
    %fprintf( 'the 0 iteration:\n' );
    %plot_cluster( clusters_new, M, mu_list_new, X );
% recluster
    T = 20;
    for t = 1:T
        
        clusters_old = clusters_new;
        mu_list_old = mu_list_new;
                
        clusters_new = group( clusters_old, mu_list_old, M );
        mu_list_new = mu_m( clusters_new, M );
        
        %fprintf( 'the %d iteration:\n', t );
        %plot_cluster( clusters_new, M, mu_list_new, X );
        keep = mean( mean( clusters_new - clusters_old ) );
        if keep == 0
            break;
        end
    end
    %fprintf(' break in %d\n', t );
    plot_cluster( clusters_new, M, mu_list_new, X );
    %ret = clusters_new;
    ret = mu_list_new;
return ;
%%
function [mu_list, clusters] = mu_m0( clusters, M )
    mu_list = rand( M, 2 );
return;

%%
function [mu_list, clusters] = mu_m1( clusters, M )
    
    % compute the nu and give clusters a label
    [row, last] = size( clusters );
    
    %nbins = floor( row / M );
    nbins = floor(  row / M );
    %head = 1 - nbins;
    %tail = 0;
    
    nbin = nbins * 2 * rand( 1, 1 );
    nbin = floor( nbin );
    head = 1;
    tail = nbin;

    mu_list = [];
    for m = 1 : M
        
        %head = head + nbins;
        %tail = tail + nbins;
        
        
        %dim = last - 2;
        dim = last - 1;
        if head == tail
            mu = clusters( head:tail, 1:dim );
            clusters( head:tail, last ) = m;
        elseif m == M
            mu = mean( clusters( head:row, 1:dim ) );
            clusters( head:row, last ) = m;
        else
            mu = mean( clusters( head:tail, 1:dim ) );
            clusters( head:tail, last ) = m;
        end
        mu_list = [ mu_list; mu ];
        
        nbin = nbins * 2 * rand( 1, 1 );
        nbin = floor( nbin );
        head = head + nbin;
        tail = tail + nbin;
    end
return;

%%
function mu_list = mu_m( clusters, M )
    
    % compute the nu and give clusters a label
    [row, last] = size( clusters );
    
    bins = zeros( M, 2 );
    binnum = 1;
    count = 0;
    %mu_list = [];
    for m = 1 : row
        xdim = 2; 
        bins(binnum, 1:2) = bins(binnum, 1:2) + clusters(m, 1:2);
        count = count + 1;
        
        if m ~= row
            nextbin = clusters(m+1, xdim+1);
            if binnum ~= nextbin
                bins( binnum, 1 ) = bins( binnum, 1 ) ./ count;
                bins( binnum, 2 ) = bins( binnum, 2 ) ./ count;
                count = 0;
                binnum = binnum + 1;
            end
        
        else
            bins( binnum, 1 ) = bins( binnum, 1 ) ./ count;
            bins( binnum, 2 ) = bins( binnum, 2 ) ./ count;
        end
    end
    mu_list = bins;
    %mu_list = [ mu_list; mu ];
return;
%%
function clusters = group( clusters, mu_list, M )
    % re group the clusters data
    [row, last] = size( clusters );
    
    for n = 1 : row     
        %dim = last - 2;
        dim = last - 1;
        minlen = 10000;
        
        node = clusters( n, 1:dim );
        for m = 1 : M
            len = norm( node - mu_list(m) );
            if( len < minlen )
                minlen = len;
                label = m;
            end
        end
        clusters( n, last ) = label;
    end
    
    % sort by the same last symbol which mean group 
    [old, idx] = sort( clusters(:, last) );
    clusters = clusters(idx, :);
return;

%%
function plot_cluster( clusters, M, mu_list, data )
%   only plot 2-dim datas
    [row, last] = size( clusters );
    
    nbins = floor( row / M );
    head = 1 - nbins;
    tail = 0;
    
    %flag = 0;
    plot( data(:, 1), data(:, 2), 'ow' );
    hold on;
    r = [1.0, 0.0, 0.0];
    g = [0.0, 1.0, 0.0];
    b = [0.0, 0.0, 1.0];
    c_list = [r; g; b];
    for n = 1:M
        %if n == 2
        %    hold on;
        %end
        head = head + nbins;
        tail = tail + nbins;
        if (n == M)
            tail = row;
        end
        
        %c = rand(3, 1);
        %cc = mod(n*n, 1.0);
        %c = [cc; cc; cc];
        
        c = c_list(n, 1:3);
        
        fprintf( 'the means %d is (%f, %f)\n', n, mu_list(n, 1), mu_list(n, 2));
        plot( mu_list(n, 1), mu_list(n, 2), '*', 'color', c );       
        %if flag == 0
        %    hold on;
        %    flag = 1;
        %end
        
        plot( clusters( head:tail, 1 ), clusters( head:tail, 2 ), '.', 'color', c );
    end
    hold off;
    title('K-means result');
    xlabel( 'x-corrdinate' );
    ylabel( 'y-corrdinate' );
    %outputpath = '/home/master/97/r97032/htdocs/show/kmeans_result.png';
    outputpath = 'outputs/kmeans.png';
    saveas( gcf, outputpath, 'png' );
return;
