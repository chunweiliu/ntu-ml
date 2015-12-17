function verErrorRs = verfinal_medium()
fold = 5;
fid = fopen('outputs/ver_medium.txt', 'w');
%step1: read data
file_data = load('dataset/proj_medium_train.dat');
Data = file_data(:, 3:end);
Labels = file_data(:, 2);
Labels = Labels*2 - 1;

% Step2: constructing weak learner
NodeNums = [1, 5, 10, 20, 40];
MaxIter = 100;
verErrorRs = zeros(length(NodeNums),1);
box = length(Labels)/fold;
for n = 1:length(NodeNums)
    NodeNum = NodeNums(n);
    weak_learner = tree_node_w(NodeNum); % pass the number of tree splits to the constructor
    
    head = 1;
    tail = box;
    RandomIndex = randperm(length(Labels));
    verErrorR = 0;
    for m = 1:fold

        ri = RandomIndex(head:tail);
        ri = ri';

        % Step2: splitting data to training and control set
        ControlData   = Data(ri,:);
        ControlLabels = Labels(ri);        

        ControlData   = ControlData';
        ControlLabels = ControlLabels';
        
        ri1 = RandomIndex(1:head-1);
        ri2 = RandomIndex(tail+1:end);
        
        head = head + box;
        tail = tail + box;

        TrainData = Data(ri1,:);
        TrainData = [TrainData; Data(ri2,:)];
        TrainLabels = Labels(ri1);
        TrainLabels = [TrainLabels; Labels(ri2)];
        
        TrainData = TrainData';
        TrainLabels = TrainLabels';
        
        [MLearners MWeights ] = ModestAdaBoost(weak_learner, TrainData, TrainLabels, MaxIter);

        ResultM = sign(Classify(MLearners, MWeights, ControlData));

        ErrorR  = sum(ControlLabels ~= ResultM);
        
        verErrorR = verErrorR + ErrorR;
                
    end
    verErrorRs(n) = verErrorR / (fold*size(ControlData,2)); 
    fprintf(fid, 'Training Data = 4000, Argument (T = 100, TreeNode = %d); The %d fold corss validation error EV = %f\n', NodeNum, fold, verErrorRs(n));
end
fclose(fid);



