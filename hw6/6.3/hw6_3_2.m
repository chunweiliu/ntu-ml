function [ trainerr, testerr ] = hw6_3_2()
train = load( 'hw6_train.dat' );
test = load( 'hw6_test.dat' );
xdim = 2;

x = train(:, 1:xdim);
y = train(:, xdim+1);
T = 100;
%h = bagginglearn( x, y, T, @stumplearn, @bootstrap );

tx = test(:, 1:xdim);
ty = test(:, xdim+1);
h = [];
for t = 1:T
    idx = bootstrap(size(x, 1));
    xt = x(idx, :);
    yt = y(idx);
    ut = 1/size(x, 1);
    
    ht = stumplearn( xt, yt, ut );
    h = [h;{ht}];
    a = ones(t,1);
    
    v = baggingfunc( @stumpfunc, h, t, a, x );
    r = (v ~= y);
    trainerr(t) = sum(r) / size(x, 1);        

    v = baggingfunc( @stumpfunc, h, t, a, tx );
    tr = (v ~= ty);
    testerr(t) = sum(tr) / size(tx, 1);
end

x = 1:1:length(trainerr);
plot(x, trainerr, 'color', 'b');
hold on;
x = 1:1:length(testerr);
plot(x, testerr, 'color', 'r');
hold off;
title('Error of Bootstrap Aggregation');
xlabel('t^t^h iteration');
ylabel('Error rate');
legend('Train err.', 'Test err.');
saveas(gcf, '6.3(2).png', 'png');
return;


