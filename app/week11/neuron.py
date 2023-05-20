import numpy as np


class Neuron:
    def __init__(self):
        self.W = None
        self.m = None
        self.n_iter = None
        self.alpha = None

    def __init___(self):
        pass

    def call(self, X, y=None, training=False, alpha=0.03, n_iter=1000):
        # no activation function
        # regression
        # gradient descent
        self.alpha = alpha
        self.n_iter = n_iter
        self.m = X.shape[0]
        y = y.reshape(-1,1)
        if training:
            # do feedforward
            # update weights
            X = np.concatenate([np.ones(self.m)[:, np.newaxis], X], axis=1)
            self.W = np.random.uniform(size=X.shape[1]).reshape(-1, 1)
            for _ in range(self.n_iter):
                y_pred = X @ self.W
                self.W -= 2 * self.alpha * X.T @ (y_pred - y) / self.m
        else:
            # do feedforward
            pass


X = np.array([[5, 4], [3, 2]])
y = np.array([2, 3])

neuron = Neuron()
neuron.call(X, y, True)
