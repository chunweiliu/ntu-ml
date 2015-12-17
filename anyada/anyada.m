function [w, h, trainerr, validerr, testerr] = anyada(x, y, T, vx, vy, tx, ty, learn, func)
N = size(x, 1);
w = zeros(T, 1);
d = ones(N, 1) ./ N;
h = cell(T, 1);

trainerr = [];
validerr = [];
testerr = [];

for t=1:T
  t
  tic,
  h{t} = learn(x, y, d);
  toc
  
  hx = func(h{t}, x);

  tic,
  w(t) = fminsearch(@(wt) (sum(repmat(d, 1, length(wt)) .* ...
                        exp(- (y .* hx) .* wt))), 0);
  toc
  d = d .* exp(- (y .* hx) * w(t));
  d = d ./ sum(d);
  
  [yy, mgn] = aggregate(w, h, func, x);  
  trainerr(t) = sum(yy ~= y) ./ N * 100;  
  [yy, mgn] = aggregate(w, h, func, vx);  
  validerr(t) = sum(yy ~= vy) ./ length(vy) * 100;
  [yy, mgn] = aggregate(w, h, func, tx);  
  testerr(t) = sum(yy ~= ty) ./ length(ty) * 100;  
end

  
  
