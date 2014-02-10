%% script - Random Forest matlab on fisher iris data
clear all;
close all;
clc;
%% params
ntrees = 35;

%% data
dims = [1 2];
load fisheriris.mat;
X = meas([51:end],dims);
X = fCenterSphereData(X')';
Y = [zeros(50,1);ones(50,1)];
[RBI] = fGenXvalBlockIndex(100,5);
Xtr = X(RBI~=5,:);
Xte = X(RBI==5,:);
Ytr = Y(RBI~=5);
Yte = Y(RBI==5);
%% Fit Model
Model                   = TreeBagger(ntrees,Xtr,Ytr,'method','classification','OOBVarImp','on');

%% evaluate
[YtePred,ProbOuts]      = predict(Model,Xte);
[~,Ypred]               = max(ProbOuts,[],2); % annoyingly outputs winning class labels as cells, but we can get it from the probability outputs
C                       = confusionmat(Yte,Ypred)
ClassificationError     = mean(Yte~=Ypred)
ClassificationAccuracy  = 1-ClassificationError

%% map decision regions and plot
gridsize = 15;
datasize = 45;

[mX,mY] = meshgrid(-3:0.05:3);
mX = mX(:);
mY = mY(:);
mXY = [mX mY];
Ymesh = ones(length(mXY),1);

[~,pred_label_mesh] = predict(Model,mXY);
pred_label_mesh = pred_label_mesh(:,1);
figure;hold;colormap('jet');
scatter(mXY(:,1),mXY(:,2),gridsize,pred_label_mesh,'Marker','+');

scatter(X(Y==0,1),X(Y==0,2),datasize,Y(Y==0),'Marker','o',...
    'MarkerFaceColor',[1 0 0],'MarkerEdgeColor',[1 0 0]);
scatter(X(Y==1,1),X(Y==1,2),datasize,Y(Y==1),'Marker','o',...
    'MarkerFaceColor',[0 0 1],'MarkerEdgeColor',[0 0 1]);





[~,pred_label_mesh] = predict(Model,mXY);
[~,pred_label_mesh] = max(pred_label_mesh,[],2);
%pred_label_mesh = pred_label_mesh-1;
figure;hold;colormap('jet');
scatter(mXY(:,1),mXY(:,2),gridsize,pred_label_mesh,'Marker','+');

scatter(X(Y==0,1),X(Y==0,2),datasize,Y(Y==0),'Marker','o',...
    'MarkerFaceColor',[0 1 0],'MarkerEdgeColor',[0 0 0]);
scatter(X(Y==1,1),X(Y==1,2),datasize,Y(Y==1),'Marker','o',...
    'MarkerFaceColor',[1 0 0],'MarkerEdgeColor',[0 0 0]);