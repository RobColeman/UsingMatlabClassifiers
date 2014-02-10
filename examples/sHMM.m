%% usage of HMM package
clear all;
close all;
clc;


load('C:\Data\SSVEPAT_Su12_03-Oct-2012_Data12.mat')
x = detrend(data.EEG(setdiff(1:64,[13 19]),:)');
x = fCenterSphereData(x')';

environment antwave64
subject flat
gc = setdiff(1:64,[13 18]);
clim = [-1 1];

% standard settings:
M = [1 1 1];
dim = size(x,2);
rvar = 1;
stopcrit.maxiters = 50;
stopcrit.minllimpr = 1e-5;
reg = 0.00001;

% initialize HMM
hmm0 = hmminit(M,dim,rvar);
% learning
tic
hmm1 = hmmem(x,hmm0,stopcrit,reg);
toc

MOG = [];
for st = 1:3
    for g = 1:M(st)
    tt = hmm1.pdf{st,1}.mean(g,:)';
    topo(tt,gc,clim);gcf;title(sprintf('state %d gaussian %d',st,g));
    end
end
  
[path,fullpath,sortofprob,scale] = hmmviterbi(x,hmm1);
