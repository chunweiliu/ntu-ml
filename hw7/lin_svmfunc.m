function fx = lin_svmfunc( theta, w, x )
%LIN_SVMFUNC: Linear SVM function 
%   INPUTS: theta -- constant term of hyperplane
%           w -- hyperplane
%           x -- testing example
%   OUTPUTS:fx -- f of x of testing example

fx = x * w - theta;