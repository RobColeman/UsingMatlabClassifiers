%% script - Random Forest matlab on fisher iris data
clear all;
close all;
clc;
%% params
ntrees = 60;

%% data

load knnClassify3c.mat;
Xtr = Xtrain;
Xte = Xtest;
Ytr = ytrain;
Yte = ytest;

%% Fit Model
Ypred                   = classify(Xte,Xtr,Ytr,'linear');
C                       = confusionmat(Yte,Ypred)
%% evaluate
ClassificationError    = mean(Yte~=Ypred)
ClassificationAccuracy = 1-ClassificationError

%% map decision regions and plot


[mX,mY] = meshgrid(-4:0.05:8);
mX = mX(:);
mY = mY(:);
mXY = [mX mY];
Ymesh = ones(length(mXY),1);

[pred_label_mesh] = classify(mXY,Xtr,Ytr,'linear');
figure;hold;
colormap('jet');
scatter(mXY(:,1),mXY(:,2),2,pred_label_mesh,'Marker','+');
scatter(Xte(Yte==1,1),Xte(Yte==1,2),25,Yte(Yte==1),'Marker','o',...
    'MarkerFaceColor',[0 0 1],'MarkerEdgeColor',[0 0 1]);
scatter(Xte(Yte==2,1),Xte(Yte==2,2),25,Yte(Yte==2),'Marker','o',...
    'MarkerFaceColor',[0 1 0],'MarkerEdgeColor',[0 1 0]);
scatter(Xte(Yte==3,1),Xte(Yte==3,2),25,Yte(Yte==3),'Marker','o',...
    'MarkerFaceColor',[1 0 0],'MarkerEdgeColor',[1 0 0]);


%%