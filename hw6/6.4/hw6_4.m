function [ trainerr, testerr ] = hw6_4()
train = load( 'hw6_train.dat' );
test = load( 'hw6_test.dat' );
xdim = 2;

T = 100;
x = train(:, 1:xdim);
y = train(:, xdim+1);
tx = test(:, 1:xdim);
ty = test(:, xdim+1);

[xnum, xdim] = size(x);
u = ones(xnum, 1) ./ xnum;


hw = [];
alp = [];
for t = 1:T
    ht = stumplearn(x, y, u);

    vt = stumpfunc(ht, x);
    rt = (vt ~= y);
    et = sum(u .* rt) / sum(u);
    
    at = 0.5 * log( (1-et) / (et) );
        
    u = u .* exp( y .* (-at) .* stumpfunc(ht, x) );
    U(t) = sum(u); 
    
    tt = t;
    alp = [alp; at];
    hw = [hw; ht];    
    g = [{tt}, {alp}, {hw}];
    
    v = adafunc(g, @stumpfunc, x);
    r = (v ~= y);
    trainerr(t) = sum(r) / size(x, 1);
    
    v = adafunc(g, @stumpfunc, tx );
    tr = (v ~= ty);
    testerr(t) = sum(tr) / size(tx, 1);
end

x = 1:1:length(trainerr);
plot(x, trainerr, 'color', 'b');
hold on;
plot(x, testerr, 'color', 'r');
plot(x, U, 'color', [.4,.7,.7]);
hold off;
title('Error of Adaptive Boosting');
xlabel('t^t^h iteration');
ylabel('Error rate');
legend('Train err.', 'Test err.', 'U');
saveas(gcf, '6.4.png', 'png');
return;
