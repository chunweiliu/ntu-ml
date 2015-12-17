function error = plr_testing( x, y, W )
%PLR_TESTING Summary of this function goes here
%   Detailed explanation goes here
xnum = size(x, 1);
xdim = size(x, 2);
error = 0;
X = [-1*ones( size(x, 1), 1), x ]; 
for n = 1:xnum
    if y(r) ~= sign( X(r, :) * W )
        error = error + 1;
    end
end
error = error / xnum;
return;
