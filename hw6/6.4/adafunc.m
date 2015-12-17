function hx = adafunc( h, func, x )
T = cell2mat(h(1));
a = cell2mat(h(2));
hh = cell2mat(h(3));
s = 0;
for t = 1:T
    ht(1) = hh(t,1);
    ht(2) = hh(t,2);
    ht(3) = hh(t,3);

    s = s + a(t) * func( ht, x );
end
hx = sign( s );
return;