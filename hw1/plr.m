clear all
% load every file we need
load data/hw1_3_train.dat

load train.txt
load test.txt
load w.txt

load train_n_0.txt
load test_n_0.txt
load w_n_0.txt

load train_n_1.txt
load test_n_1.txt
load w_n_1.txt

% assign valuable
t = train(:, 1)

v1 = train(:, 2)
plot( t, v1 )
title( '1.3(1)' )
xlabel( 't-th iteration' )
ylabel( 'training error rate ')
saveas( gcf, '1.3(1).png')

p1 = test(:, 2)
plot( t, p1 )
title( '1.3(2)' )
xlabel( 't-th iteration' )
ylabel( 'testing error rate ')
saveas( gcf, '1.3(2).png' )

a = w(1000, 3)
b = w(1000, 4)
c = w(1000, 2)
x = 0.0:0.01:1.0
h = (-a*x + c) / b
plot(x, h)
title( '1.3(3)' )
xlabel( 'x-coordinate' )
ylabel( 'y-coordinate' )

hold on
for i = 1:100,
	if hw1_3_train(i, 3) == 1
		plot( hw1_3_train(i, 1), hw1_3_train(i, 2), 'o')
	else
		plot( hw1_3_train(i, 1), hw1_3_train(i, 2), 'x')
	end
end
hold off
saveas( gcf, '1.3(3).png' )

v2 = train_n_0(:, 2)
plot( t, v2 )
title( '1.4(1)-1' )
xlabel( 't-th iteration' )
ylabel( 'training error rate ')
saveas( gcf, '1.4(1)-1.png' )

p2 = test_n_0(:, 2)
plot( t, p2 )
title( '1.4(1)-2' )
xlabel( 't-th iteration' )
ylabel( 'testing error rate ')
saveas( gcf, '1.4(1)-2.png' )

v3 = train_n_1(:, 2)
plot( t, v3 )
title( '1.4(2)-1' )
xlabel( 't-th iteration' )
ylabel( 'training error rate ')
saveas( gcf, '1.4(2)-1.png' )

p3 = test_n_1(:, 2)
plot( t, p3 )
title( '1.4(2)-2' )
xlabel( 't-th iteration' )
ylabel( 'testing error rate ')
saveas( gcf, '1.4(2)-2.png' )

