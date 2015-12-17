function stochastic_gradient_descent( trainfile , testfile )
%   STOCHASTIC GRADIENT DESCENT
%   Loading the training & test file
%   Data Format (x1, x2, y)
train = load( trainfile );
test = load( testfile );

Y = train(:, 3);
x = train(:, 1:2);
tmp = size(x);
num_col = tmp(1, 1);
c = -1* ones(num_col, 1);
X = [x, c];

%   Initial a (d+1) dimensional vector w
W = zeros(3, 1);
T = 2000;
alpha = 0.001;
nu = [];
pi = [];
fo = [];
flag = 1; % flag control 2.4(2) = 1 ; 2.4(3) = 0
for n = 1: T
    %random step
    if( flag == 1)
        r = 1 + floor(rand * (num_col)); % [0 100]
        W = W - alpha * dE_n( W, X(r, 1:2), Y(r) ) * [X(r, 1:2), -1]';
    else
    %fixed-step
        count = 0;
        for m = 1: max(size(Y))
            count = count + dE_n( W, X(m, 1:2), Y(m) ) * [X(m, 1:2), -1]';
        end
        count = count / max(size(Y));
        W = W - alpha * count;
    end

    %   TRAINING ERROR
    nu = [nu, error_count( W, train )];
    
    %   TESTING ERROR
    pi = [pi, error_count( W, test )];
end

%   Plot the figure
x = 1:T;
plot(x, nu, 'b');
xlabel('t-th iteration');
ylabel('training error rate');
if( flag == 1)
    title('2.4(2)-1');
    saveas(gcf, '2.4(2)-1.png', 'png');
else
    axis([1, 2000, 0.05, 0.2]);
    title('2.4(3)-1');
    saveas(gcf, '2.4(3)-1.png', 'png');
end

plot(x, pi, 'r');
xlabel('t-th iteration');
ylabel('testing error rate');
if( flag == 1)
    title('2.4(2)-2');
    saveas(gcf, '2.4(2)-2.png', 'png');
else
    axis([1, 2000, 0.05, 0.2]);
    title('2.4(3)-2');
    saveas(gcf, '2.4(3)-2.png', 'png');
end

hold off;
x = 0:1;
y = (-W(1,1).*x + W(3, 1)) ./ W(2, 1);
plot(x, y, 'k');
hold on;
for n = 1:max(size(train))
    if train(n,3) == 1
        plot( train(n, 1), train(n, 2), 'ro');
    else
        plot( train(n, 1), train(n, 2), 'bx');
    end
end
hold off;
xlabel('x_0-coordinate');
ylabel('x_1-coordinate');
if( flag == 1)
    title('2.4(2)-3');
    saveas(gcf, '2.4(2)-3.png', 'png');
else
    title('2.4(3)-3');
    saveas(gcf, '2.4(3)-3.png', 'png');
end


x = 0:1;
y = (-W(1,1).*x + W(3, 1)) ./ W(2, 1);
plot(x, y, 'k');
hold on;
for n = 1:max(size(test))
    if test(n,3) == 1
        plot( test(n, 1), test(n, 2), 'ro');
    else
        plot( test(n, 1), test(n, 2), 'bx');
    end
end
hold off;
xlabel('x_0-coordinate');
ylabel('x_1-coordinate');
if( flag == 1)
    title('2.4(2)-4');
    saveas(gcf, '2.4(2)-4.png', 'png');
else
    title('2.4(3)-4');
    saveas(gcf, '2.4(3)-4.png', 'png');
end



%   Generate gradient of E_n
function [ ret ] = dE_n( W, x, y )
w = W(1:2);
theta = W(3);
ret = exp(-y * (x * w - theta) ) * (-y) / (1 + exp(-y * (x * w - theta)));

function [ error ] = error_count(W, file)
count = 0;
num_col = max(size(file));
x = file(:, 1:2);
c = -1* ones(num_col, 1);
X = [x, c];
for m = 1: num_col
    y = file(m, 3);
    if y ~= sign(X(m,:) * W)
       count = count + 1;
    end
end
error = count / num_col;
