function [MLearners MWeights t_hyp] = final_medium(flag, MLearners, MWeights, t_hyp, iter)

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
NodeNum = 20;
weak_learner = tree_node_w(NodeNum); % pass the number of tree splits to the constructor

if flag == 0
    MaxIter = 100; 
    [MLearners MWeights t_hyp] = ModestAdaBoost(weak_learner, TrainData, TrainLabels, MaxIter);
else
    MaxIter = iter; 
    [MLearners MWeights t_hyp] = ModestAdaBoost(weak_learner, TrainData, TrainLabels, MaxIter, MWeights, MLearners, t_hyp);
end

% Step5: testing
file_data = load('dataset/proj_test_m.dat');
TestData = file_data';

% Step6: evaluating on control set
ResultM = (sign(Classify(MLearners, MWeights, TestData)) + 1) / 2;

% Step7: output file
fid = fopen('outputs/medium.txt','w');
fprintf(fid, '%d\n', ResultM);
fclose(fid);