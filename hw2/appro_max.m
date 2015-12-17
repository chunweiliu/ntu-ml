function appro_max()
theta = 0: 0.01 :2*pi;
y = foo( theta );
plot(theta, y);
hold on;
minimum = min(y);
fprintf('The minimum is about %f\n', minimum);
plot(1.05,minimum,'s'); % Plot at minimum of f
text(1.05,minimum-2,'Minimum');
xlabel('\theta');
ylabel('E(\theta)');
title('2.2(5)-c');
hold off;
saveas(gcf, '2.2(5).png','png');

function [ ret ] = foo( theta )
u = 0.5 * sin( theta );
v = 0.5 * cos( theta );
ret = exp(u) + exp(2.*v) + exp(u.*v) + u.*u - 3.*u.*v + 4.*v.*v - 3.*u - 5.*v;
