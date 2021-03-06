function [YPRED,a,b,alpha] = fAdaBoostedLogRegression(Xtrain,Ytrain,Xtest,Ytest,Nlearners,reg,maxiter,bagging,baggingratio,plotstuff)

%   [YPRED,a,b,alpha] = fAdaBoostedLogRegression(Xtrain,Ytrain,Xtest,Ytest,Nlearners,bagging)
%
%   Run Adaboosted Logistic Regression Classifiers
%
%   Inputs:
%       Xtrain: Training Dataset.       Row observations
%       Ytrain: Training labels.        Row observations 1,2,...
%       Xtest:  Testing observations.   Row observations
%       Ytest:  Testing Labels.         Row observations
%       Nlearners:  Number of weak learners to train.
%       bagging: 1/0 use bagging on training data to
%       baggingratio:   Ratio of trainingcasesperweaklearner/totaltrainingcases
%
%   Outputs:
%       YRPED:  Predicted lables for Xtest -1, 1
%       a:      Regression vectors per learner.     (vector,learner)
%       b:      Intercept term per learner.         (intercept,learner)
%       alpha:  Weights per learner.

if exist('param','var') == 0
    param = [];
end
if exist('bagging','var') == 0
    bagging = 0;
end

[Ntrain,D] = size(Xtrain);
[Ntest,D]  = size(Xtest);
w = ones(Ntrain,1);
tr_error = nan(1,Nlearners);
te_error = nan(1,Nlearners);

for j = 1:Nlearners
    %% bagging procedure
    if bagging == 1
        [Xbagtrain,Xbagtest,Ybagtrain,Ybagtest] = fparsedataTrainTest(Xtrain,Ytrain,baggingratio);
        [Nbagtrain,D]  = size(Xbagtrain);
        w = ones(Nbagtrain,1);
    else
        Xbagtrain = Xtrain;
        Ybagtrain = Ytrain;
    end
    %% train learners
    [a(:,j), b(:,j), Y_predi(:,j)] = fLogisticReg(Xbagtrain, Ybagtrain, w, reg, maxiter);  % train new learner
    
    Etri        = sum(Y_predi ~= sign(repmat(Ybagtrain,1,j)-1.5));             % All missclassified per learner
    Etr         = sum(Etri);                                                % All missclassified this round
    alpha       = ((ones(1,length(Etri))-(Etri./Etr))./Etr)';               % calc learner weights
    tr_error(j) = Etr/(Ntrain*j);                                           % collect training error on each trial
    
    %% test learners and apply weights to predictions
    
    
    B           = repmat(b,Ntest,1);                                        % gen intercept matrix
    YPREDi      = sign(Xtest*a-B);                                          % gen individual predictions
    YPRED       = sign(YPREDi*alpha);
    Etei        = sum(YPRED ~= sign(Ytest-1.5));
    Ete         = sum(Etei);
    te_error(j) = Ete/(Ntest*j);                                              % calc testing error: Count incorrect / (numtestobs * numlearners at iteration)
    
end

if exist('plotstuff','var') == 1
    if plotstuff == 1
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
    end
end