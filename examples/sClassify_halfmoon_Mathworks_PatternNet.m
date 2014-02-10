clear all;
close all;
clc;


load halfmoon.mat;

%figure;scatter(X(:,1),X(:,2),[],Y);

Ytrv = zeros(size(Ytr,1),2);
Ytrv(Ytr==1,1) = 1;
Ytrv(Ytr==2,2) = 1;

net = patternnet(10);

%net.trainFcn          = 'trainbr';
%netTest = train(net,Xtr',Ytrv');
% [~,Y_pred] = max(round(netTest(Xte')));
% Y_pred = Y_pred';

netTest = train(net,Xtr',Ytr','useGPU','only','showResources','yes');
Y_pred = round(netTest(Xte'))';

Err = mean(Y_pred~=Yte)