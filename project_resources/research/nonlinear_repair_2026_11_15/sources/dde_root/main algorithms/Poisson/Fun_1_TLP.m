function [f] = Fun_1_TLP(X, A1_long, pi, pen, tau) 
% return penalized loglik
% currently, pen is a scalar

f = @loglike;
    function [ll] = loglike(x) 
        penalty = pen * TLP(x, tau);

        lam = x * A1_long'; % linear combination 
        loglik = -sum(exp(lam), 1) *sum(pi,1)' + sum(X*lam .* pi, 'all');
        
        ll = penalty - loglik;
    end
end