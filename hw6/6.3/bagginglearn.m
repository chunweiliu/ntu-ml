function g = bagginglearn( x, y, T, learner, sampling )
%BAGGING 
%   Bootstrap Aggregation
h = [];
for t = 1:T
    idx = sampling(size(x, 1));
    xt = x(idx, :);
    yt = y(idx);
    ut = 1/size(x, 1);
    ht = learner( xt, yt, ut );
    h = [h;{ht}];
end
g = h;
return;
