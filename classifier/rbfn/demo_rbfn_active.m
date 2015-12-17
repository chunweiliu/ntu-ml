function a = demo_rbfn_active()
train = load( 'hw4_train.dat' );
x = train(:, 1:2);
y = train(:, 3);
k = 3;
sigma = 0.1;
a = rbfn(x, y, k, sigma);
return;