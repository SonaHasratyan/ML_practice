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
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


class DenseLayer:
    def __init__(
        self, units: int = 1, activation: str = "identity", last_layer: bool = False
    ):
        self.units = units
        self.activation = activation
        self.last_layer = last_layer
        self.derivative = 1
        self.next_layer_derivative = 1
        self.lr = 0.00003

        self.m = None

        self.X = None
        self.y = None
        self.W = None

        self.X_new = None

    def call(self, X: np.ndarray, y: np.ndarray, lr: float, is_train: bool = True):
        self.X = X
        self.m = self.X.shape[0]

        if not self.last_layer:
            self.X = np.concatenate([np.ones(self.m)[:, np.newaxis], self.X], axis=1)
            self.units += 1

        if is_train:
            self.y = y
            self.lr = lr

            self.W = np.random.rand(self.units, self.X.shape[1])

            if self.last_layer:
                self.derivative = 2 * (self.X @ self.W.T - self.y)

        self.X_new = self.X @ self.W.T

    def backpropagation(self):
        self.__calc_derivative()
        self.W -= (self.lr * np.sum(self.derivative, axis=0)).reshape(-1, 1) / self.m

    def __calc_derivative(self):
        self.derivative = self.X_new * self.next_layer_derivative[:, 0].reshape(-1, 1)


class DenseNetwork:
    def __init__(
        self,
        network: list,
        is_regression: bool = True,
        n_iter: int = 100,
        lr: float = 0.00003,
    ):
        self.network = network
        self.is_regression = is_regression
        self.n_iter = n_iter
        self.lr = lr

        self.X = None
        self.y = None

        self.n_layers = len(self.network)

    def call(self, X: np.ndarray, y: np.ndarray = None, is_train: bool = True):
        self.X = X

        if not is_train:
            self.network[0].call(X, y, self.lr, is_train)

            for _ in range(self.n_iter):
                for i in range(1, self.n_layers):
                    self.network[i].call(
                        self.network[i - 1].X_new, y, self.lr, is_train
                    )

            return self.network[-1].X_new

        self.y = y

        self.network[0].call(X, y, self.lr)

        for _ in range(self.n_iter):
            for i in range(1, self.n_layers):
                self.network[i].call(self.network[i - 1].X_new, y, self.lr)

            for i in range(1, self.n_layers):
                self.network[
                    self.n_layers - i - 1
                ].next_layer_derivative = self.network[self.n_layers - i].derivative
                self.network[self.n_layers - i - 1].backpropagation()

        return self.network[-1].X_new


np.random.seed(78)

X, y = make_regression(n_samples=1000, n_features=4, n_informative=3, n_targets=1)
y = y.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=78)

scaler = MinMaxScaler()
scaler.fit(X_train)
scaler.transform(X_train)
scaler.transform(X_test)

model = DenseNetwork(
    [
        DenseLayer(2),
        DenseLayer(3),
        DenseLayer(1, last_layer=True),
    ]
)

model.call(X_train, y_train, is_train=True)
y_pred = model.call(X_test, is_train=False)
print(mean_squared_error(y_pred, y_test))
print(r2_score(y_pred, y_test))
