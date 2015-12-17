function [w, trainerr, testerr] = hw7_3()

train = load('hw7_3_1_train.dat');
%train = load('hw7_3_2_train.dat');
x = train(:, 1:2);
y = train(:, 3);

test = load('hw7_3_1_test.dat');
%test = load('hw7_3_2_test.dat');
tx = test(:, 1:2);
ty = test(:, 3);

[xnum, xdim] = size(x);

w = [];
C = [0.01, 0.1, 1, 10, 100];
for n = 1:length(C)
    c = C(n);
    v = svmlearn_softlin( x, y, c );
    w = [w; v(1:1+xdim)'];
    
    fx = lin_svmfunc( v(1), v(2:2+xdim-1), x );
    s = sign(fx);
    t = (s ~= y);
    trainerr(n) = sum(t) / size(x, 1);
        
    %xi = v(2+xdim:length(v));
    %t = ((y .* fx + xi - 1) <  0);
    %nsv(n) = sum(t) / size(x, 1);
    
    fx = lin_svmfunc( v(1), v(2:2+xdim-1), tx );
    s = sign(fx);
    t = (s ~= ty);
    testerr(n) = sum(t) / size(tx, 1);
    
    draw_result(v, tx, ty, n, testerr, c);
    
    
end

function draw_result( v, tx, ty, n, testerr, c )
%graphic----------------
    %g(1) = 0;% non-theta
    g(1) = -v(1); 
    g(2) = v(2);
    g(3) = v(3);
    [X,Y] = meshgrid(0:.1:1, 0:.1:1);
    Z = g(1) + g(2).*X + g(3).*Y;
    contourf( X, Y, Z, [0, 0], 'LineStyle', '-', 'LineWidth', 2);
    hold on;
    s = ['.', 'x'];
    cr = ['r', 'b'];
    for m = 1:length(ty)
        plot( tx(m,1), tx(m,2), s( (ty(m)+3)/2 ), 'MarkerSize', 6,...
            'MarkerFaceColor', cr( (ty(m)+3)/2 ),...
            'MarkerEdgeColor', cr( (ty(m)+3)/2 ));
    end
    hold off;
    st = sprintf('Test result with error = (%.3f), where c = (%.2f)',...
                testerr(n), c);
    title(st);
    xlabel('x_1');
    ylabel('x_2');
    sw = sprintf('\\theta = %f, \\omega=(%f, %f)',...
                v(1), v(2), v(3));
    text(0.55, -0.1, sw);
    sf = sprintf('outputs/%d.png', n);
    saveas(gcf, sf, 'png');
    %-----------------------
return;