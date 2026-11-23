function [f] = F_1_SAEM(Xj, A1_sample_long, N, K1, C) 
% for fixed j

f = @loglike;
    function [ll] = loglike(x)
        eta = reshape(sum(bsxfun(@times, A1_sample_long, reshape(x, [1 K1 + 1 1])), 2), [N C]);
        ll = -(sum(eta,2)' * Xj - sum(exp(eta), 'all'))/C;
    end
end