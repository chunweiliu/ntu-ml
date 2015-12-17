function hx = baggingfunc( func, h, T, a, x )
h = cell2mat(h);
s = 0;
for t = 1:T
    ht(1) = h(t,1);
    ht(2) = h(t,2);
    ht(3) = h(t,3);

    s = s + a(t) * func( ht, x );
end
hx = sign( s );
return;

