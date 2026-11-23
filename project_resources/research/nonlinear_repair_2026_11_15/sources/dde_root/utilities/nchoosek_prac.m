function [y] = nchoosek_prac(n, k)



y = exp(sum(log((n-k+1):n)) - sum(log(1:k)));



end