function w = svmlearn_softlin( x, y, c )
%SOFTLIN_SVMLEARN soft margin linear SVM learning
%   INPUTS: x -- training pattern
%           y -- training label
%   OUTPUTS:w -- [t, omega, cosin]
%           t -- theta of hyperplane
%           omega -- a hyperplane with maximum margin
%           xi -- error capacity in soft margin

[xnum, xdim] = size(x);

a = 0;
aa = eye(xdim);
aaa = zeros(xnum);
H = blkdiag(a, aa, aaa);

f = [0; zeros(xdim,1); c*ones(xnum,1)];

r = zeros(xnum,1);
rr = zeros(xnum, xdim);
rrr = eye(xnum);
R = [-y, diag(y)*x, eye(xnum)];
RR = [r, rr, rrr];
R = [R; RR];
A = -R;

q = [ones(xnum,1); zeros(xnum,1)];
b = -q;

options = optimset('LargeScale', 'off', 'MaxIter', 400);
w = quadprog(H, f, A, b, [], [], [], [], [], options);
