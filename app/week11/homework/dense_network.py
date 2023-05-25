"""
    Implement DenseNetwork, using only libraries that you knew before this week
    (without tf, pytorch or similar frameworks). The good way to do it would be
    to have a DenseLayer class, objects of which (e.g., DenseLayer(64), DenseLayer(128))
    would be passed to the network during initialization. You are not restricted to use
    only call method, as we have done during the class, but you can do however you feel easy.
    Your network should include feedforward and backpropagation functionalities of dense network.
    Keep in mind, that gradient of the next layer should be used while changing weights in the
    current layer. The basic implementation can include no activation functions and be done for
    the regression task.
    Here are some points to make your task more interesting and harder,
    if this seems too easy for you:

    2.1 Add activation functions for each layer (relu and sigmoid would be good)
    2.2 Make an option for network to solve both classification and regression problems. For this
    you will just need to change loss function and its gradient
    2.3 Implement dropout logic
    2.4 Any other functionality from the lecture that you find interesting to add
"""

import numpy as np
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error


# todo delete v
class Neuron:
    def __init__(self, alpha: float = 0.03, n_iter: int = 1000, bias: bool = False):
        self.W = None
        self.m = None
        self.alpha = alpha
        self.n_iter = n_iter
        self.next_neuron_derivative = 1

    def call(self, X: np.ndarray, y: np.ndarray = None, training: bool = False):
        self.m = X.shape[0]

        if training:
            y = y.reshape(-1, 1)

            self.W = np.random.uniform(size=X.shape[1]).reshape(-1, 1)
            for _ in range(self.n_iter):
                y_pred = X @ self.W
                self.W -= (2 * self.alpha * X.T @ (y_pred - y)) / self.m
        else:
            return X @ self.W


class DenseLayer:
    def __init__(
        self, units: int = 1, activation: str = "identity", last_layer: bool = False
    ):
        self.units = units + 1
        self.activation = activation
        self.last_layer = last_layer
        self.derivative = 1
        self.next_layer_derivative = 1
        self.lr = 0.03

        self.m = None
        self.neurons = np.array(self.units)

        self.__init_neurons()

        self.X = None
        self.y = None
        self.W = np.random.rand(self.m, self.units)
        self.X_new = None

    def call(self, X, y, lr):
        self.X = X
        # todo: make y last_layer exclusive
        self.y = y
        self.lr = lr

        self.m = self.X.shape[0]

        self.X = np.concatenate([np.ones(self.m)[:, np.newaxis], self.X], axis=1)
        XW = self.X @ self.W
        if self.last_layer:
            self.derivative = 2 * (XW - self.y)
            self.X_new = (XW - self.y) ** 2
        else:
            self.derivative = XW @ self.next_layer_derivative
            self.X_new = XW

        self.W -= (self.lr * self.derivative) / self.m
        # todo: we are here

    def __init_neurons(self):
        self.neurons[0] = Neuron(bias=True)

        for i in range(1, self.units):
            self.neurons[i] = Neuron()


class DenseNetwork:
    def __init__(
        self,
        network: list,
        is_regression: bool = True,
        n_iter: int = 100,
        lr: float = 0.03,
    ):
        self.network = network
        self.is_regression = is_regression
        self.n_iter = n_iter
        self.lr = lr

        self.X = None
        self.y = None

    def call(self, X: np.ndarray, y: np.ndarray):
        self.X = X
        self.y = y

        self.network[0].call(X, y)
        for i in range(1, len(self.network)):
            self.network[i].call(self.network[i - 1].X_new, y, self.lr)
            self.network[i - 1].next_layer_derivative = self.network[i].derivative


model = DenseNetwork(
    [
        DenseLayer(2),
        DenseLayer(3),
        DenseLayer(4, last_layer=True),
    ]
)
