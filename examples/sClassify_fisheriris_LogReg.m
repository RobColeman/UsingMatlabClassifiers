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
Y = [0*ones(50,1);ones(50,1)];
[RBI] = fGenXvalBlockIndex(100,5);
Xtr = X(RBI~=5,:);
Xte = X(RBI==5,:);
Ytr = Y(RBI~=5);
Yte = Y(RBI==5);
%% Fit Model
model                   = mnrfit(Xtr,Ytr);
Probs                   = mnrval(model,Xte);
[Prob,Ypred]            = max(Probs,[],2);
C                       = confusionmat(Yte,Ypred)
%% evaluate
ClassificationError     = mean(Yte~=Ypred)
ClassificationAccuracy  = 1-ClassificationError

%% map decision regions and plot
gridsize = 15;
datasize = 45;

[mX,mY] = meshgrid(-2.5:0.05:2.5);
mX = mX(:);
mY = mY(:);
mXY = [mX mY];
Ymesh = ones(length(mXY),1);


%%

probsmesh = mnrval(model,mXY);
pred_label_mesh = probsmesh(:,1);

figure;hold;
colormap('jet');
scatter(mXY(:,1),mXY(:,2),gridsize,pred_label_mesh,'Marker','+');

scatter(X(Y==1,1),X(Y==1,2),datasize,Y(Y==1),'Marker','o',...
    'MarkerFaceColor',[1 0 0],'MarkerEdgeColor',[1 0 0]);
scatter(X(Y==2,1),X(Y==2,2),datasize,Y(Y==2),'Marker','o',...
    'MarkerFaceColor',[0 0 1],'MarkerEdgeColor',[0 0 1]);