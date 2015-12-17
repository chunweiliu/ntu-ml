function h = stumplearn(x, y, u)
[xnum, xdim] = size(x);
dirs = [1, -1];
grad_min = inf;
% design theta
%theta = -1:0.01:1;    
for dim = 1:xdim
    % design theta
    x_d = x(:, dim);
    x_s = sort(x_d);
    theta = 0.5 * (x_s(1:xnum-1) + x_s(2:xnum));
    theta = [-theta', theta'];
    for dir = dirs        
        for th = theta
            hx = sign( x_d .* dir - th );
            foo = (y ~= hx);    % 1 or 0
            grad = sum(u .* foo);
            if( grad < grad_min )
                dir_min = dir;
                dim_min = dim;
                theta_min = th;
                grad_min = grad;
            end
        end
    end
end

h = [];
h(1) = dir_min;
h(2) = dim_min;
h(3) = theta_min;
return;
