%% script - Random Forest matlab on fisher iris data
clear all;
close all;
clc;
%% params
dims2use    = [2 4];   % dimensions of fiser iris data to use, select exactly 2 for visualization
NBlocks     = 5;       % for parsing training/testing
Block2use   = 2;       % block to use for testing
ntrees      = 100;     % number of trees to grow
%% load data





load fisheriris.mat;

[COEFF, SCORE, LATENT] = PRINCOMP(X)



X = meas(:,dims2use);
Y = [ones(50,1);2*ones(50,1);3*ones(50,1)];
N = length(Y);




%% parse data to training and testing
RandomOrder = randperm(N);
NperBlock   = ceil(N/NBlocks);

OrderedBlockIndex = [];
for j = 1:(NBlocks-1)
    BlockNum = j;OrderedBlockIndex = [OrderedBlockIndex; BlockNum*ones(NperBlock,1) ];
end
BlockNum = BlockNum+1;
% Last Block
OrderedBlockIndex = [OrderedBlockIndex; BlockNum*ones(N-length(OrderedBlockIndex),1)];
RBI = OrderedBlockIndex(RandomOrder);
TrIDX = RBI ~= Block2use;
TeIDX = RBI == Block2use;
Xtr = X(TrIDX,:);
Xte = X(TeIDX,:);
Ytr = Y(TrIDX,:);
Yte = Y(TeIDX,:);

%% Fit Model
Model                   = TreeBagger(ntrees,Xtr,Ytr,'method','classification','OOBVarImp','on');
[YtePred,ProbOuts]      = predict(Model,Xte);
[~,Ypred]               = max(ProbOuts,[],2); % annoyingly outputs winning class labels as cells, but we can get it from the probability outputs
C                       = confusionmat(Yte,Ypred)
%% evaluate
ClassificationError    = mean(Yte~=Ypred)
ClassificationAccuracy = 1-ClassificationError

%% map decision regions and plot

if length(dims2use) == 2
    [mX,mY] = meshgrid(0:0.05:8);
    mX = mX(:);
    mY = mY(:);
    mXY = [mX mY];
    Ymesh = ones(length(mXY),1);
    
    [~,pred_label_mesh] = predict(Model,mXY);
    [~,pred_label_mesh] = max(pred_label_mesh,[],2);
    figure;hold;
    colormap('jet');
    scatter(mXY(:,1),mXY(:,2),2,pred_label_mesh);
    scatter(X(Y==1,1),X(Y==1,2),10,Y(Y==1),'Marker','o');
    scatter(X(Y==2,1),X(Y==2,2),10,Y(Y==2),'Marker','o');
    scatter(X(Y==3,1),X(Y==3,2),10,Y(Y==3),'Marker','o');
end