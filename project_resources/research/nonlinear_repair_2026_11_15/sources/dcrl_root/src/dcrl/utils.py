import numpy as np


def binary(x, k=None):
    x = np.array(x).reshape(-1, 1)
    if k is None:
        k = int(np.max(np.floor(np.log2(x)) + 1)) if np.all(x > 0) else 1
    else:
        kmax = int(np.max(np.floor(np.log2(np.max(x))) + 1)) if np.max(x) > 0 else 1
        assert k >= kmax, "k must be greater than or equal to the maximum binary length of x"
    divs = np.floor(x / (2 ** np.arange(k - 1, -1, -1))).astype(int)
    r = divs - np.hstack((np.zeros((len(x), 1)), 2 * divs[:, :-1]))
    return r


def TLP(x, tau):
    return np.sum(np.minimum(np.abs(x), tau))


def thres(X, tau):
    X_th = X.copy()
    index = np.abs(X) < tau
    X_th[index] = 0
    return X_th


def nchoosek_prac(n, k):
    numerator = np.sum(np.log(np.arange(n - k + 1, n + 1)))
    denominator = np.sum(np.log(np.arange(1, k + 1)))
    return np.exp(numerator - denominator)


def initialize_function():
    def zero_function(x):
        return 0
    return zero_function


def sigmoid(x):
    return 1 / (1 + np.exp(-x))
