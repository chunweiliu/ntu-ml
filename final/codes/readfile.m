function readfile()
fid1 = fopen('dataset/proj_test.dat','r');
fid2 = fopen('dataset/proj_test_m.dat','w');
for n = 1:10000
    fscanf(fid1, '%s', 2);
    num = fscanf(fid1, '%f', 78);
    num = num';
    fprintf(fid2, '%f ', num);
    fprintf(fid2, '\n');
end
fclose(fid1);
fclose(fid2);

