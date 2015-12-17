function result()
x = [1, 5, 10, 20, 40];
sm100 = [.3332,.333,.3358,.34,.3382];
md100 = [.3276,.2996,.3,.2938,.2942];
lg100 = [.3366,.3082,.2932,.288,.2872];

sm1000 = [.3332,.342,.3384,.3422,.3366];
md1000 = [.3276,.2956,.2978,.2938,.2944];
lg1000 = [.3366,.3026,.2954,.2832,.284];

plot(x,sm1000, x,md1000, x,lg1000, 'LineStyle', '-', 'LineWidth', 2);
hold on;
plot(x,sm100,x,md100,x,lg100, 'LineStyle', '-.', 'LineWidth', 1);


legend('sm','md','lg');
title('AdaBoost Result on Competition');
xlabel('Depth of Decision Tree');
ylabel('Testing Error');
saveas(gcf,'outputs/adaresult.png','png');
