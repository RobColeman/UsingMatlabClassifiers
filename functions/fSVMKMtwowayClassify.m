function [muerr err] = fSVMKMtwowayClassify(data,labels,ratio,iter,c,epsilon,kernel,kerneloption,verbose)
% 
%
%   [muerr err] = fSVMKMtwowayClassify(data,labels,ratio,iter,c,epsilon,kernel,kerneloption,verbose)
%
%       Prepare date and make call to  SVMKM toolbox's SVMCLASS.m for Two way
%       SVM model generation and SVMVAL.m for classification with built model.
%
%       Inputs:
%           DATA:           Data matrix with observations along rows.
%           LABELS:         Data labels, column vector with cases along rows.
%                           Labels must be [-1 1].
%           RATIO:          Ratio of Training/All Observations for parsing data into
%                           Training cases and testing cases.
%           ITER:           Number of iterations to run model building and
%                           classification.   Data is shuffled and parsed and new
%                           model is built every iteration.
%           C:              Default: inf
%           EPSILON:        regularization factor for matrix inversions
%           KERNEL: String: Kernel to use. Default: 'gaussian'
%           KERNELOPTION:   kernel option.  For Gaussian/RBF this is Kernelwidth
%           VERBOSE:        print to command window durring operation
%       Outputs:
%           MUERR:          Mean classification error across iterations
%           ERR:            Vector of Classification errors per iteration
%
%
%
%
%



for i = 1:iter
    [xapp,xtest,yapp,ytest] = fparsedataTrainTest(data,labels,ratio);
    %c = inf;
    %epsilon = .000001;
    %kerneloption= 10000;
    %kernel='gaussian';
    %verbose = 0;
    [xsup,w,b,pos]=svmclass(xapp,yapp,c,epsilon,kernel,kerneloption,verbose);
    % b = 0; % clear bias?
    ypred = svmval(xtest,xsup,w,b,kernel,kerneloption);
    err(i) = mean(abs(sign(sign(ypred)-ytest))); % calculate classification error on each run
end
muerr = mean(err);

