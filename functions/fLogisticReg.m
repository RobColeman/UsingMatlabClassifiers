% [a, b, Y_pred] = fLogisticReg(X, Y, w, reg, param)
%
%   Logistic regression.
%
%       Inputs:
%           X: input data
%           Y: Class Labels. Two classes (1,2)
%           W: weights on training samples.  Wector of length == number of
%                                                                samples.
%           REG: regularization term to be added to diagonal of covariance
%                for inversion.
%               Default: 1e-5
%           PARAM: Structure variable with fields.
%               PARAM.MAXITER (an iteration limit)
%               PARAM.EPSILON (used to test convergence)
%               PARAM.RANDINIT (used to randomly initialize the weight
%                               vector)  PARAM.RANDINIT*rand(D,1);
%       Outputs:
%           A: Regression coefficients
%           B: Intercept for regression
%           Y_pred: Predicted Labels for X
%
%       Model is  E(Y) = 1 ./ (1+exp(-A*X))
%           Classify with: sign(X*a - b)
%
%       Birthed by Robert Coleman  2011/01/08
%       Revised                    2011/02/19
%
%                  Contact: Robert Coleman
%                     colemanr (at) uci (dot) edu

function [a, b, Y_pred] = fLogisticReg(X, Y, w, reg, maxiter)

% process parameters

[n, d] = size(X);
%% catch inputs
if ((nargin < 3) || (isempty(w)))
    w = ones(n, 1);
end

if ((nargin < 4) || (isempty(reg)))
    reg = 1e-5;
end


if (length(reg) == 1)
    regMAT = speye(d) * reg;
elseif (length(reg(:)) == d)
    regMAT = spdiags(reg(:), 0, d, d);
else
    error('reg weight vector should be length 1 or %d', d);
end

if (~exist('maxiter','var'))
    maxiter = 200;
end

if (~exist('epsilon','var'))
    epsilon = 1e-7;
end

a = rand(d,1);

% do the regression
oldexpY = -ones(size(Y));
for iter = 1:maxiter
    
    adjY = X * a;
    expY = 1 ./ (1 + exp(-adjY));
    deriv = expY .* (1-expY);
    wadjY = w .* (deriv .* adjY + (Y-expY));
    weights = spdiags(deriv .* w, 0, n, n);
    
    a = inv(X' * weights * X + regMAT) * X' * wadjY;
    
    if (sum(abs(expY-oldexpY)) < n*epsilon)
        Y_pred = X*a;
        b = mean(Y_pred);
        Y_pred = sign(Y_pred - b);
        return;
    end
    oldexpY = expY;
end
Y_pred = X*a;
b = mean(Y_pred);
Y_pred = sign(Y_pred - b);

