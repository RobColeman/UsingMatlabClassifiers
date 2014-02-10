function ConfMat = fConfustionMatrix(Ytest,Ypred)
%
%   Usage: ConfMat = fConfustionMatrix(Ytest,Ypred)
%
%   Classes must be integers 1:Nc
%
%
%
%
%
Nc = max(Ytest);
ConfMat = nan(Nc);

for C1 = 1:Nc
    CyteIDX = Ytest==C1;
    for C2 = 1:Nc
        ConfMat(C1,C2) = sum(Ypred(CyteIDX)==C2);   
    end
end % over class labels
end % function