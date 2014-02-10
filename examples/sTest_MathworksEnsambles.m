% clear all;
% close all;
% clc;
%
% nlearn      = 100;
% learners    = 'tree';
% fractionforlearner = 0.2;
% method      = 'AdaBoostM1';
% type        = 'classification';
%
%
% load halfmoon.mat;
%
% %figure;scatter(X(:,1),X(:,2),[],Y);
%
%
% % ens = fitensemble(Xtr,Ytr,method,nlearn,learners,'type',type,'crossval','on',...
% %     'fresample',fractionforlearner,'replace','on');
%
%
% ens = fitensemble(Xtr,Ytr,'AdaBoostM1',100,'Tree')
% Y_pred = predict(ens,Xte);
%
% Err = mean(Y_pred~=Yte)
%
% figure;scatter(Xte(:,1),Xte(:,2),[],Y_pred);
% [Y_pred Yte]
%


%% ionosphere data

clear all;
close all;
clc;

load ionosphere;

N = length(Y);
NBlocks = 11;
ntrees  = 300;

[RBI,NBlocks] = fGenXvalBlockIndex(N,NBlocks);

for ii = 1:size(Y);if strcmpi(Y{ii},'b');T(ii,1) = 1;else T(ii,1) = 2;end;end;

for xv = 1:NBlocks
    Xtr = X(RBI~=xv,:);
    Xte = X(RBI==xv,:);
    Ytr = T(RBI~=xv,:);
    Yte = T(RBI==xv,:);
    
    ens                 = fitensemble(Xtr,Ytr,'AdaBoostM1',ntrees,'Tree');
    y_predENS           = predict(ens,Xte);
    
    model               = train(Ytr,sparse(Xtr));
    y_predSVM           = predict(Yte,sparse(Xte),model);
    
    Model               = TreeBagger(ntrees,Xtr,Ytr,'method','classification');
    [YtePred,ProbOuts]  = predict(Model,Xte);
    [~,y_predRF]        = max(ProbOuts,[],2);
    
    ErrENS(xv) = mean(y_predENS~=Yte);
    ErrSVM(xv) = mean(y_predSVM~=Yte);
    ErrRF(xv)  = mean(y_predRF~=Yte);
end

mean(ErrENS)
mean(ErrSVM)
mean(ErrRF)
