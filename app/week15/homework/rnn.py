"""
    The task is to train a next character prediction model by Vanilla RNN. As of training dataset you can select any
    text you want (book may be an easy and efficient solution). Model structure (input, weights, output etc.) should
    be as discussed during the practice.
"""

import numpy as np
from sklearn.metrics import mean_squared_error


class RNNBlock:
    def __init__(self, f_W="tanh", state_size=32, learning_rate=0.03):

        self.f_W = f_W
        self.state_size = state_size
        self.f_Ws = {None: Linear(), "sigmoid": Sigmoid(), "relu": ReLu(), "tanh": TanH()}
        self.f_W = self.f_Ws[self.f_W]
        self.learning_rate = learning_rate

        self.x = None
        self.h_prev = 0  # h_prev is the h_0 at the beginning
        self.y_pred = None
        self.h_next = None
        self.loss = None
        self.grad = None
        self.next_grad = None

        self.W_hx = None
        self.W_hh = 0.01 * np.random.randn(self.state_size, self.state_size)
        self.W_hy = None
        self.bias_h = 0
        self.bias_0 = 0

    def feedforward(self, x, y=None):
        self.x = x
        self.y = y

        if not self.W_hh:
            self.__init_weights()

        self.h_next = self.f_W(self.W_hh @ self.h_prev + self.W_hx @ self.x + self.bias_h)
        self.y_pred = self.W_hy @ self.h_next + self.bias_0
        self.loss = mean_squared_error(self.y_pred, self.y)

    def backpropagation(self):
        # todo
        pass

    def __init_weights(self):
        x_shape = self.x.shape
        if len(x_shape) == 2:
            x_shape = x_shape[1]

        self.W_hx = 0.01 * np.random.randn(self.state_size, x_shape)
        self.W_hy = 0.01 * np.random.randn(self.state_size, self.y.shape)


class VanillaRNN:
    def __init__(self, X, y, batch_size=None, n_iter=1000):
        self.X = X
        self.y = y
        self.n = self.X.shape[1]
        self.n_iter = n_iter

        self.rnn_layers = []

        self.batch_size = batch_size if batch_size else self.n

    def train(self):
        for i in range(self.n):
            self.rnn_layers.append(RNNBlock("tanh", state_size=8, learning_rate=0.03))

        self.rnn_layers[0].feedforward(self.X[:, 0])
        for _ in range(self.n_iter):
            for i in range(1, self.n):
                for j in range(i, i + self.batch_size):
                    self.rnn_layers[i].h_prev = self.rnn_layers[i - 1].h_next
                    self.rnn_layers[j].feedforward(self.X[:, j])
                    # todo

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
        return np.cosh(X) ** (-2)
