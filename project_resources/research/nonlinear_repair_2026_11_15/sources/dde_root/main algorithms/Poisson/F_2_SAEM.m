function [f] = F_2_SAEM(A1_sample_k, A2_sample_long, N, K2, C) 
f = @loglike;
    function [ll] = loglike(x) 
        eta = reshape(sum(bsxfun(@times, A2_sample_long, reshape(x, [1 K2 + 1 1])), 2), [N C]);
        ll = -(sum(A1_sample_k .* eta, 'all') - sum(log(1+exp(eta)), 'all'))/C;
    end
end