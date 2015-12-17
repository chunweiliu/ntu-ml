function mediumdata()
file_data = load('dataset/proj_medium_train.dat');
Data = file_data(:, 3:end);
Labels = file_data(:, 2);
Labels = Labels*2 - 1;

Data = Data';
Labels = Labels';

TrainLabels = Labels; 
% Step2: splitting data to training and control set
 % Step5: testing
file_data = load('dataset/proj_test_m.dat');
TestData = file_data';

TrainDataB   = Data;
TrainDataB([29,55,47,48,49,50,51],:) = [];
TestDataB = TestData;
TestDataB([29,55,47,48,49,50,51],:) = [];

TrainDataC   = Data;
TrainDataC([29,55,47,48,49,50,51,20,21,22,44,45,46],:) = [];
TestDataC = TestData;
TestDataC([29,55,47,48,49,50,51,20,21,22,44,45,46],:) = [];

TrainDataD   = Data;

for n = 1:size(TrainDataD,2)
    for m = 1:size(TrainDataD,1)
        if m == 20 && TrainDataD(m,n) > 900
            TrainDataD(m,n) = 0;
        elseif m == 21 && TrainDataD(m,n) > 900
            TrainDataD(m,n) = 0;
        elseif m == 22 && TrainDataD(m,n) > 900
            TrainDataD(m,n) = -0.62;
        elseif m == 44 && TrainDataD(m,n) > 900
            TrainDataD(m,n) = 0.976;
        elseif m == 45 && TrainDataD(m,n) > 900
            TrainDataD(m,n) = 0.841;
        elseif m == 46 && TrainDataD(m,n) > 900
            TrainDataD(m,n) = -0.613;
        elseif m == 44 && TrainDataD(m,n) < 0.001
            TrainDataD(m,n) = 0.976;
        elseif m == 45 && TrainDataD(m,n) < 0.001
            TrainDataD(m,n) = 0.841;
        elseif m == 46 && TrainDataD(m,n) > -0.000000001
            TrainDataD(m,n) = -0.613;
        end
    end
end
TrainDataD([29,55,47,48,49,50,51],:) = [];
TestDataD = TestData;
TestDataD([29,55,47,48,49,50,51],:) = [];

TrainDataE   = Data;

for n = 1:size(TrainDataE,2)
    for m = 1:size(TrainDataE,1)
        if m == 20 && TrainDataE(m,n) > 900
            TrainDataE(m,n) = 0;
        elseif m == 21 && TrainDataE(m,n) > 900
            TrainDataE(m,n) = 0;
        elseif m == 22 && TrainDataE(m,n) > 900
            TrainDataE(m,n) = -0.62;
        elseif m == 44 && TrainDataE(m,n) > 900
            TrainDataE(m,n) = 0.431081;
        elseif m == 45 && TrainDataE(m,n) > 900
            TrainDataE(m,n) = 0.371226;
        elseif m == 46 && TrainDataE(m,n) > 900
            TrainDataE(m,n) = -0.27054;
        end
    end
end
TrainDataE([29,55,47,48,49,50,51],:) = [];
TestDataE = TestData;
TestDataE([29,55,47,48,49,50,51],:) = [];


% Step3: constructing weak learner
MaxIter = 100;
NodeNum = 20;
TrainDatas = {TrainDataB, TrainDataC, TrainDataD, TrainDataE};
TestDatas = {TestDataB, TestDataC, TestDataD, TestDataE};

for n = 1:length(TrainDatas)
    TrainData = cell2mat(TrainDatas(n));
    TestData = cell2mat(TestDatas(n));
    
    weak_learner = tree_node_w(NodeNum); % pass the number of tree splits to the constructor

        
    [MLearners MWeights] = ModestAdaBoost(weak_learner, TrainData, TrainLabels, MaxIter);   

    % Step6: evaluating on control set
    ResultM = (sign(Classify(MLearners, MWeights, TestData)) + 1) / 2;

    % Step7: output file
    filepath = sprintf('outputs/mediumdata_n%d_t%d_d%d.txt', NodeNum, MaxIter,n);
    fid = fopen(filepath,'w');
    fprintf(fid, '%d\n', ResultM);
    fclose(fid);
end