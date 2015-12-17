function fxt = svmfunc( alpha, x, y, c, kernel, p, tx )
%SVMFUNC Kernel SVM function
%   INPUTS: freealpha -- lagrangean multiplier of those
%           kernel -- kernel function
%           tx -- testing example
%   OUTPUTS:fx -- f of x of testing example

% get theta
thetas = zeros(size(x, 1), 1);
for n = 1:size(x, 1)
    s = 0;
    for m = 1:size(x, 1)
        s = s + alpha(m) * y(m) * kernel(x(n,:), x(m,:), p);
    end
    thetas(n) = s - 1/y(n);
end
epslion = 1e-5;
idx = find((alpha > epslion) & (alpha < c - epslion));

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

%syms txn
fxt = zeros(size(tx, 1), 1);
for n = 1:size(tx, 1)    
    txn = tx(n, :); 
    
    f = 0;    
    for m = 1:size(sv, 1)
        tmpsv = sv(m, :);
        
        k = kernel( tmpsv, txn, p );
        tmp = (salpha(m) * sy(m) * k);
        f = f + tmp;
    end
        
    fxt(n) = f - theta;
end
%ezpolt('fxt');