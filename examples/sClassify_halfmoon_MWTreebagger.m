%% script - Random Forest matlab on fisher iris data
clear all;
close all;
clc;
%% params
ntrees = 100;

%% data
dims = [1 2];
load fisheriris.mat;
X = meas([51:end],dims);
X = fCenterSphereData(X')';
Y = [ones(50,1);2*ones(50,1)];
[RBI] = fGenXvalBlockIndex(100,4);
Xtr = X(RBI~=4,:);
Xte = X(RBI==4,:);
Ytr = Y(RBI~=4);
Yte = Y(RBI==4);
%% Fit Model
Model                   = TreeBagger(ntrees,Xtr,Ytr,'method','classification','OOBVarImp','on');
[YtePred,ProbOuts]      = predict(Model,Xte);
[~,Ypred]               = max(ProbOuts,[],2); % annoyingly outputs winning class labels as cells, but we can get it from the probability outputs
C                       = confusionmat(Yte,Ypred)
%% evaluate
ClassificationError    = mean(Yte~=Ypred)
ClassificationAccuracy = 1-ClassificationError

%% map decision regions and plot
gridsize = 15;
datasize = 45;

[mX,mY] = meshgrid(-3:0.05:3);
mX = mX(:);
mY = mY(:);
mXY = [mX mY];
Ymesh = ones(length(mXY),1);

[~,pred_label_mesh] = predict(Model,mXY);
[~,pred_label_mesh] = max(pred_label_mesh,[],2);
pred_label_mesh = pred_label_mesh-1;
figure;hold;colormap('jet');
scatter(mXY(:,1),mXY(:,2),gridsize,pred_label_mesh,'Marker','+');

scatter(X(Y==1,1),X(Y==1,2),datasize,Y(Y==1),'Marker','o',...
    'MarkerFaceColor',[0 0 1],'MarkerEdgeColor',[0 0 1]);
scatter(X(Y==2,1),X(Y==2,2),datasize,Y(Y==2),'Marker','o',...
    'MarkerFaceColor',[0 1 0],'MarkerEdgeColor',[0 1 0]);