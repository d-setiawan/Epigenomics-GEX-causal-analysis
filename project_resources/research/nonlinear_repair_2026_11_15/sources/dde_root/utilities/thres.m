function X_th = thres(X, tau)
    index = abs(X) < tau;
    X_th = X;
    X_th(index) = 0;
end