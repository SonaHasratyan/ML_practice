import numpy as np


class Activation:
    def __call__(self, X):
        pass

    def derivative(self, X):
        pass


class Sigmoid(Activation):
    def __call__(self, X):
        return 1 / (1 + np.exp(-X, dtype=np.float128))

    def derivative(self, X):
        return X * (1 - X)


class ReLu(Activation):
    def __call__(self, X):
        return (X >= 0) * X

    def derivative(self, X):
        return X >= 0


class Linear(Activation):
    def __call__(self, X):
        return X

    def derivative(self, X):
        return 1


class TanH(Activation):
    def __call__(self, X):
        # return (np.exp(X) - np.exp(-X)) / (np.exp(X) + np.exp(-X))
        return np.tanh(X)

    def derivative(self, X):
        return np.cosh(X) ** (-2)
