function [phi_1, phi_2] = hw7_1_1b()
x = [1 0; 0 1; 0 -1; -1 0; 0 2; 0 -2; -2 0];
y = [-1; -1; -1; 1; 1; 1; 1];

x1 = x(:, 1);
x2 = x(:, 2);

phi_1 = x2 .^2 - 2*x1 - 1;
phi_2 = x1 .^2 - 2*x2 + 1;


g(1) = -0.5;
g(2) = 1;
g(3) = 0;
margin = 0.5;
[X,Y] = meshgrid(-5:.2:5,-5:.2:5);
Z = g(1) + g(2).*X + g(3).*Y;
contourf( X, Y, Z, [0, 0], 'LineStyle', '-', 'LineWidth', 2);
hold on;
contour( X, Y, Z, [-margin, margin], 'LineStyle', ':', 'LineWidth', 2);

s = ['o', 'o'];
c = ['r', 'b'];
for n = 1:length(y)
    plot( phi_1(n), phi_2(n), s( (y(n)+3)/2 ), 'color', 'b', 'MarkerSize', 6, 'MarkerFaceColor', c( (y(n)+3)/2 ),...
        'MarkerEdgeColor', c( (y(n)+3)/2 ));
    if phi_1(n) == 0 || phi_1(n) == 1        
        plot( phi_1(n), phi_2(n), 'o', 'color', 'b', 'MarkerSize', 12  );
    end
end
hold off;

title('7.1(1)(b)');
xlabel('\phi_1');
ylabel('\phi_2');
legend('hyperplane','margin');
saveas(gcf, 'outputs/7.1(1)(b).png', 'png');
return;
