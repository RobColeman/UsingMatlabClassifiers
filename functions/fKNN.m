function yhat = fKNN(xtrain,labels,xtest,k)
%
%
%
%
%
%
%
%



xtrain = real(xtrain);
xtest = real(xtest);


[n d] = size(xtrain);
if size(xtest,1) > size(xtest,2)
    xtest = xtest';
end

XTESTMAT = repmat(xtest,n,1);

diff = sum((xtrain-XTESTMAT).^2,2);
for i = 1:k
    [~,minIDX(i)] = min(diff); %#ok<AGROW>
end
yhat = mode(labels(minIDX));

end % function
for m = 1:3
    for c = conds
    temp = squeeze(mean(XX(Y==c,:,:,3),1));
    temp = abs(centerandspheredata(temp')');
    figure;imagesc(temp);
    ttl = sprintf('Mode %d Cond %d',m,c);
    end % conditions
end % modes
