function g = demo_plr_active()
%DEMO function of machine learning algorthm
%
train = load( 'hw1_3_train.dat' );
x = train(:, 1:2);
y = train(:, 3);
T = 100;
g = plr( x, y, T );
return;

function g = plr( x, y, T )
%PERCEPTRON LEARNING ALGORITHM
% INPUTS    : plr( x, y, T ), x is data, y is label, T is #iteration
% OUTPUTS   : g( W ), W = [theta; w]

X = [-1*ones( size(x, 1), 1), x ];
W = zeros( size(X, 2), 1 );
r = floor( rand(1, 1) * (size(x, 1) - 1) + 1 );

if  y(r) ~= sign( X(r, :) * W )
    W = W + ( y(r) .* X(r, :)' );
end
[xx,Map] = draw(W, x, y, 1, 0, 0);
imwrite(xx, Map, 'demo_plr.gif', 'GIF', 'WriteMode', 'overwrite', ...
                'DelayTime',0, 'LoopCount', Inf );

for t = 2:T
    r = floor( rand(1, 1) * (size(x, 1) - 1) + 1 );
    time = 0;
    if  y(r) ~= sign( X(r, :) * W )        
        time = 2;
    end    
    
    [xx,Map] = draw(W, x, y, t, r, time);
    imwrite(xx,Map,'demo_plr.gif','GIF','WriteMode','append','DelayTime',time);
    if time == 2
        W = W + ( y(r) .* X(r, :)' );
    end
end
g = W / norm(W);
return;

function [xx, Map] = draw(g, x, y, t, r, flag)
[X,Y] = meshgrid(0:.2:1,0:.2:1);
Z = -g(1) + X*g(2) + Y*g(3);
contourf( X, Y, Z, 1 );
hold on;
m = ['x'; 'o'];
for k = 1:size(x,1)
    s = (y(k)+3)/2;
    if r == k
        if flag == 2
            plot( x(k, 1), x(k, 2), m(s), 'Color', 'y', 'MarkerSize', 12 );
        else
            plot( x(k, 1), x(k, 2), m(s), 'Color', 'w','MarkerSize', 8 );
        end
    else
        plot( x(k, 1), x(k, 2), m(s), 'Color', 'g' );
    end
end
hold off;
gname = sprintf( 'Demo of Perceptron Learning Algorithm in Iteration %d', t );
title(gname);
xlabel('x1-coordinate');
ylabel('x2-coordinate');

af = getframe(gcf);
f = af.cdata;
[xx, Map] = rgb2ind(f,256);
return;
