gammas = linspace(0.01,10,100);

for i = 1:length(gammas)
    gamma      = gammas(i);
    model       = rvmSimpleFit(Xtr,Ytr,gamma);
    rvs(i).rv   = model.relevant;
    yhat        = rvmSimplePredict(model,Xte);
    err(i)      = mean(yhat~=Yte);
end % over 


figure;plot(err);