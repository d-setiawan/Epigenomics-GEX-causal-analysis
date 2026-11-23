function [X, A1, A2] = generate_X_normal(N, prop_true, B1, B2, gamma)
% Generate data; gamma is the variance

    K2 = size(prop_true, 2);
    
    % generate A2
    A2 = zeros(N, K2);
    for k = 1:K2
        A2(:,k) = binornd(1, prop_true(k), N, 1);
    end
    
    % generate A1
    A1 = binornd(1, logistic([ones(N,1), A2] * B2'));
    
    % generate X
    X = normrnd([ones(N,1), A1] * B1', repmat(sqrt(gamma'),N,1));
end