function res = TLP(x, tau)  % need tau > 0 
    res = sum(min(abs(x), tau), 'all');
end