%% script - Random Forest matlab on fisher iris data
clear all;
close all;
clc;
%% params
ntrees = 60;

%% data

load bishop2class.mat;
rnd = randperm(200);
Xtr = X(rnd(1:150),:);
Xte = X(rnd(151:end),:);
Ytr = Y(rnd(1:150));
Yte = Y(rnd(151:end));


%% Fit Model
Model                   = TreeBagger(ntrees,Xtr,Ytr,'method','classification','OOBVarImp','on');
[YtePred,ProbOuts]      = predict(Model,Xte);
[~,Ypred]               = max(ProbOuts,[],2); % annoyingly outputs winning class labels as cells, but we can get it from the probability outputs
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

[~,pred_label_mesh] = predict(Model,mXY);
[~,pred_label_mesh] = max(pred_label_mesh,[],2);
figure;hold;
colormap('jet');
scatter(mXY(:,1),mXY(:,2),2,pred_label_mesh,'Marker','+');
scatter(X(Y==1,1),X(Y==1,2),25,Y(Y==1),'Marker','o',...
    'MarkerFaceColor',[0 0 1],'MarkerEdgeColor',[0 0 1]);
scatter(X(Y==2,1),X(Y==2,2),25,Y(Y==2),'Marker','o',...
    'MarkerFaceColor',[0 1 0],'MarkerEdgeColor',[0 1 0]);
% scatter(Xte(Yte==3,1),Xte(Yte==3,2),25,Yte(Yte==3),'Marker','o',...
%     'MarkerFaceColor',[1 0 0],'MarkerEdgeColor',[1 0 0]);


%%