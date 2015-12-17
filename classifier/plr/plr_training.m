function g = plr_training( x, y, t )
%PERCEPTRON LEARNING ALGORITHM
% INPUTS    : plr( x, y, t ); x is data, y is label, t is #iteration
% OUTPUTS   : g( W ); W = [theta; w]

X = [-1*ones( size(x, 1), 1), x ];
W = zeros( size(X, 2), 1 );
for tt = 1:t
    r = floor( rand(1, 1) * (size(x, 1) - 1) + 1 );
    if  y(r) ~= sign( X(r, :) * W )
        W = W + ( y(r) .* X(r, :)' );
    end
end

g = W / norm(W);
return;