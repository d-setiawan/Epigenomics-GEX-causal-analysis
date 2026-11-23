function [prop, B1, B2, A1_new, A2_new, t] = get_SAEM_poisson(X, prop_in, B1_in, B2_in, pen_1, pen_2, tau, A1_in, A2_in, C, tol)
%% Algorithm 2 (penalized SAEM) for Poisson-two-latent-layer DDEs
% @X: N x J count data matrix
% @prop_in, B1_in, B2_in, A1_in, A2_in: initialization for corresponding parameters
% @pen_1, pen_2, tau: tuning parameters for the TLP penalty; pen_1, pen_2 corresponds to the magnitude (lambda in the paper)
%     and tau denotes the threshold value
% @C: number of stochastic approximation samples
% @tol: tolerance for convergence

% definitions
K2 = size(prop_in,2);
[J, K1] = size(B1_in);
K1 = K1-1;

N = size(X, 1);
prop = prop_in;
B1 = B1_in;
B2 = B2_in;

% iteration settings
t = 1;                  % iteration index
iter_indicator = true;

prop_update = prop_in;
B1_update = zeros(J, K1+1);
B2_update = zeros(K1, K2+1);

% optimization settings
bb = []; Aeq = []; beq = [];
AA = [];

%% general coefs
lb_1 = [-5*ones(1,1), -2.5*ones(1, K1)]; % zeros(J, K1+1);
ub_1 = 5*ones(1, K1+1);

lb_2 = [-5*ones(1,1), -2.5*ones(1, K2)]; % zeros(J, K1+1);
ub_2 = 5*ones(1, K2+1);
options = optimset('Display', 'off', 'MaxIter', 10);

% partition_ftn = sum(log(factorial(X)), 'all');
A1_new = A1_in;
A2_new = A2_in;

A1_sample_long = zeros(N,K1+1,C);
A2_sample_long = zeros(N,K2+1,C);

ber_2 = zeros(N, K2);
ber_1 = zeros(N, K1);
q = zeros(N, K2);

% optimization functions
ftn_pen_1 = @(x) pen_1 * TLP(x(2:end), tau);
ftn_pen_2 = @(x) pen_2 * TLP(x(2:end), tau);

f_old_1 = cell(J,1);
for j = 1:J
    f_old_1{j} = @(x) 0;
end
f_old_2 = cell(K1,1);
for k = 1:K1
    f_old_2{k} = @(x) 0;
end

%% main iteration
while iter_indicator
    A1_old = A1_new;
    A2_old = A2_new;

    %% Simulation step
    for i = 1:N            
    % compute complete conditionals for A2                                
        for l = 1:K2
            A2_i = A2_old(i,:);
            A2_i(l) = 1; eta_1 = B2 * [1, A2_i]'; % K1 x 1
            A2_i(l) = 0; eta_0 = B2 * [1, A2_i]';

            ber_2(i,l) = logistic(log(prop_update(l) / (1 - prop_update(l))) + ...
                sum(A1_old(i,:)' .* B2(:,l+1)) - sum(log( (1 + exp(eta_1)) ./ (1 + exp(eta_0)))));
        end
        
        % compute complete conditionals for A1
        for k = 1:K1
            A1_i = A1_old(i,:);
            A1_i(k) = 1; eta_1 = B1 * [1, A1_i]'; % J x 1
            A1_i(k) = 0; eta_0 = B1 * [1, A1_i]';
            ber_1(i,k) = logistic([1, A2_old(i,:)] * B2(k,:)' + sum(X(i,:)' .* B1(:,k+1)) ...
               - sum(exp(eta_1)-exp(eta_0)));
        end
    end
    
    % sample each latent variable
    for c = 1:C
        A2_sample_long(:,:,c) = [ones(N,1), double(rand(N,K2) < ber_2)];
        A1_sample_long(:,:,c) = [ones(N,1), double(rand(N,K1) < ber_1)];
    end
    
    for l = 1:K2
        q(:,l) = (1 - 1/t)*q(:,l) + 1/t * ber_2(:,l);
    end
    
    %% Stochastic approximation M-step
    % prop
    prop_update = sum(q, 1) /N;

    % B1
    for j = 1:J
        Xj = X(:,j);
        f_loglik = F_1_SAEM(Xj, A1_sample_long, N, K1, C);
        f_old_1{j} = @(x) (1-1/t) * f_old_1{j}(x) + 1/t * f_loglik(x);
        f_j = @(x) f_old_1{j}(x) + ftn_pen_1(x);
        B1_update(j,:) = fmincon(f_j, B1(j,:), AA, bb, Aeq, beq, lb_1, ub_1, nonlcon, options);
    end

    % B2
    for k = 1:K1
        A1_sample_k = reshape(A1_sample_long(:,k+1,:), [N C]);
        f_loglik = F_2_SAEM(A1_sample_k, A2_sample_long, N, K2, C);
        f_old_2{k} = @(x) (1-1/t) * f_old_2{k}(x) + 1/t * f_loglik(x);
        f_k = @(x) f_old_2{k}(x) + ftn_pen_2(x);
        B2_update(k,:) = fmincon(f_k, B2(k,:), AA, bb, Aeq, beq, lb_2, ub_2, nonlcon, options);
    end

    %% compute log-lik
    err = sqrt(norm(prop - prop_update, "fro")^2 + norm(B1 - B1_update, "fro")^2 + norm(B2 - B2_update, "fro")^2);

    A1_new = double(mean(A1_sample_long(:,2:end,:), 3) > 0.5); 
    A2_new = double(mean(A2_sample_long(:,2:end,:), 3) > 0.5);

    prop = prop_update;
    B1 = [B1_update(:, 1), thres(B1_update(:,2:end), tau)];
    B2 = [B2_update(:, 1), thres(B2_update(:,2:end), tau)];
    t = t + 1;
    
    fprintf('EM Iteration %d,\t Err %1.1f\n', t, err);
    iter_indicator = ( abs(err) > tol & t < 50);
end
