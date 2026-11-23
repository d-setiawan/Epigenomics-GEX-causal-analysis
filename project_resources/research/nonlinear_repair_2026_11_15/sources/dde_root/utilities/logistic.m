function [y] = logistic(x)
% X can be a vector
    y = 1 ./(1+exp(-x));
end