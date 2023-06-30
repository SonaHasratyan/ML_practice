import numpy as np


class RNNBlock:
    def __init__(self, f_W="tanh", state_size=32):
        # todo: should we have num_hidden_statements?

        self.f_W = f_W
        self.state_size = state_size
        self.f_Ws = {None: Linear(), "sigmoid": Sigmoid(), "relu": ReLu(), "tanh": TanH()}
        self.f_W = self.f_Ws[self.f_W]

        self.x = None
        self.h_prev = None
        self.y = None
        self.h_next = None

        self.W_hx = None
        self.W_hh = np.random.randn(self.state_size, self.state_size)
        self.W_hy = None
        self.bias_h = 0  # todo: ask/discuss
        self.bias_0 = 0

    def feedforward(self, x, h_prev, y=None):
        self.x = x
        self.h_prev = h_prev
        self.y = y

        if not self.W_hh:
            self.__init_weights()

        self.h_next = self.f_W(self.W_hh @ self.h_prev + self.W_hx @ self.x + self.bias_h)
        self.y = self.W_hy @ self.h_next + self.bias_0

    def backpropagation(self):
        # todo
        pass

    def __init_weights(self):
        x_shape = self.x.shape
        if len(x_shape) == 2:
            x_shape = x_shape[1]

        self.W_hx = np.random.randn(self.state_size, x_shape)
        self.W_hy = np.random.randn(self.state_size, self.y.shape)


class VanillaRNN:
    def __init__(self, X, y, batch_size=None, n_iter=1000):
        self.X = X
        self.y = y
        self.n = self.X.shape[1]
        self.n_iter = n_iter

        self.RNNBlock = RNNBlock("tanh")

        self.batch_size = batch_size if batch_size else self.n

    def train(self):
        h_prev = 0  # h_prev is the h_0 at the beginning
        # todo: should we keep rnns in a list, especially for the backprop?

        for _ in range(self.n_iter):
            for i in range(self.n):
                batch_X = self.X[:, i:i + self.batch_size]
                for x in batch_X:
                    self.RNNBlock.feedforward(x, h_prev)
                    h_prev = self.RNNBlock.h_next

                self.RNNBlock.backpropagation()

                i += self.batch_size


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
        return np.cosh(X) ** 2
