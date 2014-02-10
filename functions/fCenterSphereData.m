function X = fCenterSphereData(X)


% centeres and spheres X
%   X = fCenterSphereData(X)
%       X observations along columns
%
%   

[d,N] = size(X);

mux = mean(X,2);
X = X - repmat(mux,[1,N]);
Hx = diag(1./std(X,[],2));
X = Hx*X;
end% function