function alpha = hw7_1_2a()
%HW7_1_2A
x = [1 0; 0 1; 0 -1; -1 0; 0 2; 0 -2; -2 0];
y = [-1; -1; -1; 1; 1; 1; 1];

alpha = svmlearn( x, y, inf, @kerpoly, 2);

thetas = zeros(size(x, 1), 1);
for n = 1:size(x, 1)
    s = 0;
    for m = 1:size(x, 1)
        s = s + alpha(m) * y(m) * kerpoly(x(n,:), x(m,:), 2);
    end
    thetas(n) = s - 1/y(n);
end
epslion = 1e-5;
idx = find((alpha > epslion) & (alpha < inf - epslion));

if isempty(idx)
    % if we don't have free support vector, random chose one
    %idx = floor(rand(1,1)*(size(x,1)-1));
    
    % of average those
    idx = 1:size(x,1);
end
freetheta = thetas(idx);
theta = mean(freetheta);

idx = find(alpha>epslion);
salpha = alpha(idx, :);
sv = x(idx, :);
sy = y(idx, :);

syms x1 x2    
f = 0;    
for m = 1:size(sv, 1)
    tmpsv1 = sv(m, 1);
    tmpsv2 = sv(m, 2);
        
    k = (1 + (tmpsv1*x1 + tmpsv2*x2)) .^ 2;
    tmp = (salpha(m) * sy(m) * k);
    f = f + tmp;
end        
fxt = f - theta;
ezplot(fxt);
hold on;
s = ['o', 'o'];
c = ['r', 'b'];
for n = 1:length(y)
    plot( x(n,1), x(n,2), s( (y(n)+3)/2 ), 'color', 'b', 'MarkerSize', 6, 'MarkerFaceColor', c( (y(n)+3)/2 ),...
        'MarkerEdgeColor', c( (y(n)+3)/2 ));
    if alpha(n) > 1e-5
        plot(  x(n,1), x(n,2), 'o', 'color', 'b', 'MarkerSize', 12  );
    end
end
hold off;

title('7.1(2)(b)');
xlabel('x_1');
ylabel('x_2');
legend('hyperplane');
saveas(gcf, 'outputs/7.1(2)(b).png', 'png');