%% script - Random Forest matlab on fisher iris data
clear all;
close all;
clc;
%% params
ntrees = 100;

%% data

load halfmoon.mat;

%% Fit Model
model                   = NaiveBayes.fit(Xtr,Ytr);
Ypred                   = model.predict(Xte);
Ypred                   = classify(Xte,Xtr,Ytr,'linear');
C                       = confusionmat(Yte,Ypred)
%% evaluate
ClassificationError    = mean(Yte~=Ypred)
ClassificationAccuracy = 1-ClassificationError

%% map decision regions and plot
gridsize = 15;
datasize = 45;

[mX,mY] = meshgrid(-2.5:0.05:2.5);
mX = mX(:);
mY = mY(:);
mXY = [mX mY];
Ymesh = ones(length(mXY),1);

pred_label_mesh     = model.predict(mXY);
figure;hold;
colormap('jet');
scatter(mXY(pred_label_mesh==1,1),mXY(pred_label_mesh==1,2),gridsize,...
    ones(sum(pred_label_mesh==1),1)*2,'Marker','+',...
    'MarkerFaceColor',[0 0 1],'MarkerEdgeColor',[0 0 1]);
scatter(mXY(pred_label_mesh==2,1),mXY(pred_label_mesh==2,2),gridsize,...
    ones(sum(pred_label_mesh==2),1)*2,'Marker','+',...
    'MarkerFaceColor',[0 1 0],'MarkerEdgeColor',[0 1 0]);




scatter(X(Y==1,1),X(Y==1,2),datasize,Y(Y==1),'Marker','o',...
    'MarkerFaceColor',[0 0 1],'MarkerEdgeColor',[0 0 1]);
scatter(X(Y==2,1),X(Y==2,2),datasize,Y(Y==2),'Marker','o',...
    'MarkerFaceColor',[0 1 0],'MarkerEdgeColor',[0 1 0]);



%%