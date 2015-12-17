function [ trainerr, testerr ] = hw6_3_1()
train = load( 'hw6_train.dat' );
test = load( 'hw6_test.dat' );
xdim = 2;

x = train(:, 1:xdim);
y = train(:, xdim+1);
u = 1/size(x, 1);
h = stumplearn( x, y, u );

v = stumpfunc( h, x );
r = (v ~= y);
trainerr = sum(r) * u;

tx = test(:, 1:xdim);
ty = test(:, xdim+1);

v = stumpfunc( h, tx );
tr = (v ~= ty);
testerr = sum(tr) / size(tx, 1);
return;
