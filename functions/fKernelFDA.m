function [Xtr_proj,EigVecs,EigVals,Ktrain,Xte_proj] = fKernelFDA(Xtrain,Ytrain,Xtest,kerneltype,kerneloption,numeigs,regfactor)

%   varargout = fKernelFDA(Xtrain,Ytrain,Xtest,Ytest,Kernel,Kerneloption,numeigs,decisiontype)
%
%       Kernel Fisher Discriminant Analysis and Projection
%
%
%       Inputs:
%           Xtrain: Training data.  Row Observations
%           Ytrain: Training Labels. Row Observations cooresponding to
%                   Xtrain.  Class labels should be integers, 1,2,...,N
%           Kernel: Kerneltype to use.  See fmakeKernel.m for instructions
%                   use 'none' to perform regular LDA
%           Kerneloption: Option for Kernel generation. See fmakeKernel for
%                   instructions.
%           numeigs: Use 1:numeigs largest eigenvectors for projection of data
%           regfactor:  Regularization term for matrix inversions
%
%       Outputs:
%           Xtr_proj:   Projection of Xtrain data along numeigs maximally
%                       seperable eigenvectors of kernel.
%           EigVecs:    All Eigenvectors from Sb/Sw - Scatter Ratio
%           EigVals:    All Eigenvalues from Sb/Sw  - Scatter Ratio
%           Ktrain:     Training Kernel, NxN size

if ~exist('regfactor','var')
    regfactor = 0;
elseif isempty(regfactor)
    regfactor = 0;
end
if ~exist('numeigs','var');
    numeigs = size(Xtrain,2);
elseif isempty(numeigs)
    numeigs = size(Xtrain,2);
end

%% Kernel Fisher Linear Discriminant Analysis
if ~exist('kerneltype','var');
    Ktrain = Xtrain;
elseif strcmp(kerneltype,'none');
    Ktrain = Xtrain;
else
    [Ktrain] = fmakeKernel(Xtrain,kerneltype,kerneloption,Xtrain);    % gen training kernel
end % check for non-kernel

%% get labels

Utags = unique(Ytrain);

%% main

mui = [];                                           % init variable
diff = mean(Ktrain(Ytrain==Utags(1),:))-mean(Ktrain(Ytrain==Utags(2),:));   %% fix for multi-class
Sb = diff'*diff;                                    % between class scatter matrix

for k=1:length(Utags)                       % within class scatter matrix
    xi=Ktrain(Ytrain==Utags(k),:);          % find within or without class cases
    eval(['Xtr',num2str(k),'=xi;']);        % add variable for Xtrn
    XI = xi'*xi;                            % within class scatter per class
    eval(['XTR',num2str(k),'=XI;']);        % add variable for XTRn
    mui(end+1,:,:)=XI;                      % fill withing class means matrix
end % over classes

Sw = squeeze(sum(mui,1));                   % sum across classes
REGMAT = regfactor*eye(length(Sw));
Sw = Sw+REGMAT;
Scatterratio = Sb/Sw;                       % ratio Scatter between / Scatter within
[Vecs,Vals] = eig(Scatterratio);            % eignevalue decomp
j = 0;
EigVals = diag(Vals);
EigVals = EigVals(1:numeigs);
EigVecs = Vecs(:,1:numeigs);

Xtr_proj = Ktrain*EigVecs;                  % project training data to LDA defined space
%%% only works for linear right now
if exist('Xtest','var')  
     Xte_proj = Xtest*EigVecs;    
end
end % function