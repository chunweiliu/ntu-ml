function [trainerr, testerr, nsv] = hw7_4_1()

train = load('hw7_4_train.dat');
x = train(:, 1:2);
y = train(:, 3);

test = load('hw7_4_test.dat');
tx = test(:, 1:2);
ty = test(:, 3);

cc = [0.001, 1, 1000];
dd = [3, 6, 9];

for n = 1:length(dd)
    d = dd(n);
    for m = 1:length(cc)
        c = cc(m);
        
        alpha = svmlearn(x, y, c, @kerpoly, d);
        
        epslion = 1e-5;
        idx = find(alpha > epslion);       
        sv = x(idx, :);        
        nsv(n, m) = length(sv) / size(x, 1);
        
        fx = svmfunc( alpha, x, y, c, @kerpoly, d, x );
        s = sign(fx);
        t = (s ~= y);
        trainerr(n, m) = sum(t) / size(x, 1);
        
        fx = svmfunc( alpha, x, y, c, @kerpoly, d, tx );
        s = sign(fx);
        t = (s ~= ty);
        testerr(n, m) = sum(t) / size(tx, 1);
        
    end
end