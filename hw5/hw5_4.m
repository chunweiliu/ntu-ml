function [nu_list, pi_list] = hw5_4()
    trainfile = 'hw5_4_train.dat';
    testfile = 'hw5_4_test.dat';
    [nu_list, pi_list] = naive_bayes( trainfile, testfile );
return;