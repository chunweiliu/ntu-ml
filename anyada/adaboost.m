function [w, h] = adaboost(x, y, T)
%ADABOOST: AdaBoost with decision stump for binary classifications
% input
%   - x: n x d double   training data
%   - y: n x 1 {-1, 1}  training labels
%   - T: 1 x 1 int      number of weak classifiers
% output
%   - w: T x 1 double   weight of the hypothesis
%   - h: T x 1 cell     hypothesis

% Initialize the learning model {w, h} and data weight d
w = zeros(T, 1);
h = cell(T, 1);
d = ones(size(x, 1), 1) ./ size(x, 1);

for t = 1:T
  % Select a weak classifier from the pool
  h{t} = stumplearn(x, y, d);
  
  % Set weight w of the classifier
  hx = stumpfunc(h{t}, x);  
  w(t) = fminsearch(@(wt) (sum(repmat(d, 1, length(wt)) .* ...
                           exp(-(y .* hx) .* wt))), 0);

  % Update the weights of data points for the next iteration
  d = d .* exp(-(y .* hx) * w(t));
  d = d ./ sum(d);
end
end

function h = stumplearn(x, y, d)
%STUMPLEARN: Learning the decision stump
% input
%   - x: n x d double   training data
%   - y: n x 1 {-1, 1}  training labels
%   - d: n x 1 double   weights of training data
% output
%   - h: 3 x 1 double   hypothesis
maxgrad = -1;
for dim = 1:size(x, 2)
  xx = x(:, dim);
  xxx = sort(xx);
  alpha = 0.5 * (xxx(1:(length(xx)-1)) + xxx(2:length(xx)));
  alpha = [-inf, alpha', inf];

  hxa = sign(repmat(xx, 1, length(alpha)) - ...
             repmat(alpha, size(xx, 1), 1));
  hxa = hxa + (hxa == 0);

  grad = sum(repmat(y .* d, 1, length(alpha)) .* hxa);
  
  [best, idx] = max(abs(grad));
  idx = idx(1);

  alpha = alpha(idx);
  dir = (grad(idx) >= 0) * 2 - 1;

  if best > maxgrad
    maxalpha = alpha;
    maxdir = dir;
    maxgrad = best;
    maxdim = dim;
  end  
end

h = [maxdir, maxalpha, maxdim]';
end

function hx = stumpfunc(h, x)
%STUMPFUNC: Predicting labels by decision stump
% input
%   - h: 3 x 1 double   hypothesis
%   - x: n x d double   training data
% output
%   - hx:n x 1 {1, -1}  predict labels by decision stump
dir = h(1);
alpha = h(2);
dim = h(3);

hx = sign((x(:, dim) - alpha));
hx = hx + (hx == 0);
hx = dir .* hx;
end