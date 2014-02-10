function  [predClass argmaxpost postloglik Models] = fRNaiveBayesClassify(Xtrain,Ytrain,Xtest,Models)
% [predClass argmaxpost postloglik] = fRNaiveBayesClassify(Xtrain,Ytrain,Xtest)
%
%       NaiveBayes Classifier
%
%       Inptus:
%           Xtrain: Training Cases observations along rows.  N x D matrix
%           Ytrain: Class Labels - Integers 1,...,n
%           Xtest: Test cases
%
%       OutPuts:
%           PredClass: Predicted Class for Xtest Cases
%           argmaxpost: Largest Posterior log liklihood for Xtest Cases
%           postloglik: Posterior log liklihoods per class for Xtest Cases
%           Models: Structure contraining the Class model parameters
%                   muCi - per class Means
%                   sigmaCi - per class Variance
%
%
%
%   Birthed by Robert Coleman on 20110122 - 
%       revised 20110217
%       revised 20110305
%       Contact: Robert Coleman  - colemanr (at) uci (dot) edu

%% Error Catch

if size(Xtrain,1) ~= size(Ytrain,1)
    error('Labels and Data obs must be same number')
end
%% Naive Bayes Algorithm - Gaussian model

if nargin < 4   % calculate centroids and independant variances
    numclasses = length(unique(Ytrain));
    [N d] = size(Xtrain);
    muCi = nan(numclasses,d);
    sigmaCi = nan(numclasses,d);
    for c=1:numclasses
        idx = find(Ytrain==c);
        muCi(c,:) = sum(Xtrain(idx,:),1)/length(idx);
        MUCI = repmat(muCi(c,:),length(idx),1);
        sigmaCi(c,:) = sum(((Xtrain(idx,:)-MUCI).^2),1)/length(idx);
    end
end

if nargin > 4 % take Input models
    muCi = Models.muCi;
    sigmaCi = Models.sigmaCi;
end

if nargin > 2    % Test Xtest
    [Ntest d] = size(Xtest);
    postloglik = nan(length(Xtest),numclasses);
    for c = 1:numclasses
        MUCI = repmat(muCi(c,:),Ntest,1);
        SIGMACI = repmat(sigmaCi(c,:),Ntest,1);
        postloglik(:,c) = sum(log((1./sqrt(2.*pi.*SIGMACI.^2)).*exp(-((Xtest-MUCI).^2)./(2.*SIGMACI.^2))),2); % calc posterior
    end
end

Models.muCi = muCi;
Models.sigmaCi = sigmaCi;

if nargin > 2 % Classify Xtest by liklihoods
    [argmaxpost predClass] = max(postloglik,[],2);
else % or return empty vectors
    argmaxpost = [];
    predClass = [];
    postloglik = [];
end