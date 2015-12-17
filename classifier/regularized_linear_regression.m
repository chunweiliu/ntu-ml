function regularized_linear_regression( trainfile , testfile )
%LINEAR_REGRESSION 
%   linear_regression( trainfile, testfile );
%   Load the training data (x1, x2, y)
train = load( trainfile );

Y = train(1:120, 3);

x = train(1:120, 1:2);
tmp = size(x);
c = -1* ones(tmp(1, 1), 1);
X = [c, x];

bins = 2:-1:-10;
nub = [];
nuv = [];
pib = [];
%for n = 1: max(size(bins))
for n = bins
    %lamda = 10 .^ bins(n);
    lamda = 10 .^ n;
    %   Linear Regression get the W( theta0, w0 )'
    [r0, c0] = size(X');
    [r1, c1] = size(X);
    W = inv( X' * X + lamda * eye(r0, c1)) * X' * Y;

    %   TRAINING ERROR ( BASE )
    nub = [nub, error_count(W, train(1:120, 1:2), train(1:120, 3))];

    %   TRAINING ERROR ( VALIDATION )
    nuv = [nuv, error_count(W, train(121:200, 1:2), train(121:200, 3))];
    
    %   TESTING ERROR
    test = load( testfile );
    pib = [pib, error_count(W, test(:, 1:2), test(:, 3))];
end
%   Plot the figure
plot(bins, nub, '-or', bins, nuv, '-+b', bins, pib, '-*k');

title('2.3(4)');
xlabel('log_1_0(\lambda)');
ylabel('error rate of \nu_b(g), \nu_v(g), \pi^{\^}(g)');
h = legend('\nu_b(g)', '\nu_v(g)', '\pi^{\^}(g)');
set(h,'Interpreter','tex');
mkdir( 'outputs/RLR' );
saveas(gcf, 'outputs/RLR/1.png', 'png');

function [ error ] = error_count(W, x, y)
%   Decision function g0(x) = sign(<w0, x> - theta0), we need to compare it
%   with y0, so g0 is a column vector;
w = W(2:3, 1);
theta = W(1, 1);
    
tmp = size(y);
error = 0;
for n = 1: tmp(1, 1)
    g = sign( x(n, 1:2) * w  - theta );
    %fprintf('This is the %d-th ( %d, %d )\n', n, g, y(n));
    if g ~= y(n)
        error = error + 1;
    end
end
error = error / tmp(1, 1);
    
    