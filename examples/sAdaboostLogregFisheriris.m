%% adaboost - logistic regression - fisher iris data

clear all;
close all;
clc;

%% params

classes = 'versicolor vs virginica';
tttratio = 3/10;        % training to testing data ratio
reg = 0.00001;          % regularization factor for matrix inversions
param.maxiter = 3;      % few iterations for each logistic regression learner
Nlearners = 50;

%% load data, parse into training and test set, buffer output
load fisheriris;

switch classes
    case 'setosa vs versicolor';
        meas = meas(1:100,:);
        Y = [ones(50,1);2*ones(50,1)];
        numclasses = 2;
    case 'setosa vs virginica';
        meas = [meas(1:50,:);meas(101:150,:)];
        Y = [ones(50,1);2*ones(50,1)];
        numclasses = 2;
    case 'versicolor vs virginica';
        meas = meas(51:150,:);
        Y = [ones(50,1);2*ones(50,1)];
        numclasses = 2;
    case 'setosa vs versicolor vs virginica';
        % meas = meas
        Y = [ones(50,1);2*ones(50,1);3*ones(50,1)];
        numclasses = 3;
end

[Xtrain,Xtest,Ytrain,Ytest] = fparsedataTrainTest(meas,Y,tttratio);
[Ntrain,D] = size(Xtrain);
[Ntest,D]  = size(Xtest);
w = ones(Ntrain,1);

for i = 1:Nlearners
    %% train learners
    
    [a(:,i), b(:,i), Y_predi(:,i)] = fLogisticReg(Xtrain, Ytrain, w, reg);  % train new learner
    
    Ei          = sum(Y_predi ~= sign(repmat(Ytrain,1,i)-1.5));             % All missclassified per learner
    E           = sum(Ei);                                                  % All missclassified this round
    alpha       = ((ones(1,length(Ei))-(Ei./E))./E)';                        % calc learner weights
    tr_error(i) = E/(Ntrain*i);                                             % collect training error on each trial

    %% test learners and apply weights to predictions
    
    B           = repmat(b,Ntest,1);                                        % gen intercept matrix
    YPREDi      = sign(Xtest*a-B);                                          % gen individual predictions
    YPRED       = sign(YPREDi*alpha);
    Ei          = sum(YPRED ~= sign(Ytest-1.5));
    E           = sum(Ei);                                                  
    te_error(i) = E/(Ntest*i);                                              % calc testing error: Count incorrect / (numtestobs * numlearners at iteration)
end


figure;hold;
subplot(2,1,1);
plot(tr_error);
title('Training error');
ylabel('err');
subplot(2,1,2);
plot(te_error);
title('Testing error');
ylabel('err');
xlabel('# of learners');
