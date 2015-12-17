function alpha = svmlearn( x, y, c, kernel, kp)
%SVMLEARN Soft-Margin kernel SVM learning algo
%   INPUTS:     x           -- training pattern
%               y           -- training label
%               c           -- constrant upper bound of alpha
%               kernel      -- kernel function of svm
%               kp          -- kernel perameter
%   OUTPUTS:    alpha       -- lagrangean multiplier

[xnum, xdim] = size(x);

H = zeros(xnum, xnum);
for n = 1:xnum
    for m = 1:xnum
        H(n, m) = y(n) * y(m) * kernel(x(n,:), x(m,:), kp);
    end
end
%H = y*y' + kernel(x, x, kp);

f = -1*ones(xnum,1);

A = [y'; -y'; eye(xnum); -1.*eye(xnum)];

b = [0; 0; c*ones(xnum,1); zeros(xnum,1)];

%Aeq = diag(y);
%Aeq = repmat(y', xnum, 1);
Aeq = [];

%beq = zeros(xnum,1);
beq = [];

%lb = zeros(xnum,1);
lb = [];

%ub = c*ones(xnum,1);
ub = [];

options = optimset('LargeScale', 'off', 'MaxIter', 1000);
alpha = quadprog(H, f, A, b, Aeq, beq, lb, ub, [], options);
