function [y] = logit(x)
% X can be a vector
    y = log(x./(1-x));
end