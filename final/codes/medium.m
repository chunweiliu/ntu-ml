function medium()
%step1: read data
file_data = load('dataset/proj_medium_train.dat');
Data = file_data(:, 3:end);
Labels = file_data(:, 2);
Labels = Labels*2 - 1;

Data = Data';
Labels = Labels';


% Step2: splitting data to training and control set
TrainData   = Data;
TrainLabels = Labels;

% Step3: constructing weak learner
MaxIter = 100;
NodeNums = [1, 5, 10, 20, 40];
for NodeNum = NodeNums
    weak_learner = tree_node_w(NodeNum); % pass the number of tree splits to the constructor

        
    [MLearners MWeights] = ModestAdaBoost(weak_learner, TrainData, TrainLabels, MaxIter);

    % Step5: testing
    file_data = load('dataset/proj_test_m.dat');
    TestData = file_data';

    % Step6: evaluating on control set
    ResultM = (sign(Classify(MLearners, MWeights, TestData)) + 1) / 2;

    % Step7: output file
    filepath = sprintf('outputs/medium_n%d_t%d.txt', NodeNum, MaxIter);
    fid = fopen(filepath,'w');
    fprintf(fid, '%d\n', ResultM);
    fclose(fid);
end