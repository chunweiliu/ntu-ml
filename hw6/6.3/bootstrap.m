function p = bootstrap( t )
%BOOTSTRAP 
for n = 1:t
    p(n) = floor(rand(1,1)*(t-1)+1);
end
p = p';
return;
