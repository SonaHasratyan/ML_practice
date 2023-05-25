import numpy as np


class Neuron:
    def __init__(self, alpha: float = 0.03, n_iter: int = 1000, bias: bool = False):
        self.W = None
        self.m = None
        self.alpha = alpha
        self.n_iter = n_iter

    def call(self, X: np.ndarray, y: np.ndarray = None, training: bool = False):
        self.m = X.shape[0]
        X = np.concatenate([np.ones(self.m)[:, np.newaxis], X], axis=1)

        if training:
            y = y.reshape(-1, 1)

            self.W = np.random.uniform(size=X.shape[1]).reshape(-1, 1)
            for _ in range(self.n_iter):
                y_pred = X @ self.W
                self.W -= (2 * self.alpha * X.T @ (y_pred - y)) / self.m
        else:
            return X @ self.W


X = np.array([[5, 4], [3, 2]])
y = np.array([2, 3])

neuron = Neuron()
neuron.call(X, y, True)
print(neuron.W)
