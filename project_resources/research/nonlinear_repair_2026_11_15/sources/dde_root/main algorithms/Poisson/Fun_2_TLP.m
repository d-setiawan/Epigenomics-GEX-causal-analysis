function [f] = Fun_2_TLP(K1, A2_long, psi, psi_2, pen, tau) 
% return penalized loglik
f = @loglike;
    function [ll] = loglike(x) 
        penalty = pen * TLP(x, tau);

        eta = x * A2_long';
        loglik = sum(eta .* psi, 'all') - sum(log(exp(eta)+1),1) * psi_2;
        ll = penalty - loglik;
    end
end