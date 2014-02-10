%% script - Random Forest matlab on fisher iris data
clear all;
close all;
clc;

%% data
load halfmoon.mat;
Y = Y-1;
%% Fit Model

model       = train(Ytr,sparse(Xtr));

[YtrPred]   = predict(Ytr,sparse(Xtr),model,'-b 1');
[Ypred,acc,prob]     = predict(Yte,sparse(Xte),model,'-b 1');

C                       = confusionmat(Yte,Ypred)
%% evaluate
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
Ynot  = zeros(length(mXY),1);


[Ypredmesh,acc,pred_label_mesh]     = predict(Ynot,sparse(mXY),model,'-b 1');
figure;hold;
colormap('jet');
scatter(mXY(:,1),mXY(:,2),gridsize,pred_label_mesh,'Marker','+');

scatter(X(Y==0,1),X(Y==0,2),datasize,Y(Y==0),'Marker','o',...
    'MarkerFaceColor',[1 0 0],'MarkerEdgeColor',[1 0 0]);
scatter(X(Y==1,1),X(Y==1,2),datasize,Y(Y==1),'Marker','o',...
    'MarkerFaceColor',[0 0 1],'MarkerEdgeColor',[0 0 1]);