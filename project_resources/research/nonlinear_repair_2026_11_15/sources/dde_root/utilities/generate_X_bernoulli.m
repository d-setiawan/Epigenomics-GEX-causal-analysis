function [X, A1, A2] = generate_X_bernoulli(N, prop_true, B1, B2)
% Generate data 

    K2 = size(prop_true, 2);
    
    % generate A2
    A2 = zeros(N, K2);
    for k = 1:K2
        A2(:,k) = binornd(1, prop_true(k), N, 1);
    end
    
    % generate A1
    A1 = binornd(1, logistic([ones(N,1), A2] * B2'));
    
    % generate X
    X = binornd(1, logistic([ones(N,1), A1] * B1'));
end