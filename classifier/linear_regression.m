function [ w, theta ] = linear_regression( trainfile, testfile )
%LINEAR_REGRESSION 
%   linear_regression( trainfile, testfile );
%   Load the training data (x1, x2, y)
train = load (trainfile);

Y = train(:, 3);

x = train(:, 1:2);
tmp = size(x);
c = -1* ones(tmp(1, 1), 1);
X = [c, x];

%   Linear Regression get the W( theta0, w0 )'
W = inv( X' * X ) * X' * Y;

%   TRAINING ERROR
nu = error_count(W, train(:, 1:2), train(:, 3));

%   TESTING ERROR
test = load (testfile);
pi = error_count(W, test(:, 1:2), test(:, 3));

fprintf('( %f, %f, %f)\n', W(1,1), W(2,1), W(3,1));
fprintf('( %f , %f )\n', nu, pi);

%   Plot the figure
x = 0:1;
y = (-W(2,1).*x + W(1, 1)) ./ W(3, 1);
plot(x, y, 'k');
hold on;
tmp = size(train);
for n = 1:tmp(1, 1)
    if train(n,3) == 1
        plot( train(n, 1), train(n, 2), 'o' ,'MarkerEdgeColor' , 'r');
    else
        plot( train(n, 1), train(n, 2), 'x', 'MarkerEdgeColor' , 'b');
    end
end
hold off;
title('2.3(2)');
xlabel('x_0-coordinate');
ylabel('x_1-coordinate');
mkdir( 'outputs/LR' );
saveas(gcf, 'ouputs/LR/1.png', 'png');

function [ error ] = error_count(W, x, y)
%   Decision function g0(x) = sign(<w0, x> - theta0), we need to compare it
%   with y0, so g0 is a column vector;
w = W(2:3, 1);
theta = W(1, 1);
    
tmp = size(y);
error = 0;
for n = 1: tmp(1, 1)
    g = sign( x(n, 1:2) * w  - theta );
    if g ~= y(n)
        error = error + 1;
    end
end
error = error / tmp(1, 1);
    
    