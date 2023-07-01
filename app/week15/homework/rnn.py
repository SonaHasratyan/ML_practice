"""
    The task is to train a next character prediction model by Vanilla RNN. As of training dataset you can select any
    text you want (book may be an easy and efficient solution). Model structure (input, weights, output etc.) should
    be as discussed during the practice.
"""

# import numpy as np  # relatively imported from activation
import string

from sklearn.metrics import mean_squared_error
from activation import *

from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(sparse=False)
oheables = np.array([i for i in string.ascii_letters + string.digits + string.punctuation + " " + "\n"]).reshape(-1, 1)
ohes = ohe.fit_transform(oheables)
ohe_dict = {}
for i in range(oheables.shape[0]):
    ohe_dict.update({oheables[i][0]: ohes[i]})
# print(ohe_dict)

with open("dracula.txt") as dracula:
    lines = dracula.readlines()

removables = []
for i in range(len(lines)):
    lines[i] = lines[i].strip()
    # lines[i] = lines[i].translate(str.maketrans('', '', string.punctuation))
    if not lines[i] or lines[i] == "":
        removables.append(i)

removables = np.array(removables)
for i in range(len(removables)):
    del lines[removables[i]]
    removables = removables - 1

text = ""
for line in lines:
    add_at_beginning = "\n" if line[0].isupper() else " "
    text += add_at_beginning + line
text = text.strip()

data = [i for i in text]

data_ohe = [ohe_dict[i] for i in data]
input_size = 5
data = data_ohe

dracula_X = [data[i:i + input_size] for i in range(len(data) - input_size + 1)]
dracula_y = dracula_X[1:]
dracula_X = dracula_X[:-1]

# print(dracula_X[0], dracula_y[0], sep="\n")
# print(dracula_X[-1], dracula_y[-1], sep="\n")


class RNNBlock:
    def __init__(self, f_W="tanh", state_size=32, learning_rate=0.03):
        self.f_W = f_W
        self.state_size = state_size
        self.f_Ws = {
            None: Linear(),
            "sigmoid": Sigmoid(),
            "relu": ReLu(),
            "tanh": TanH(),
        }
        self.f_W = self.f_Ws[self.f_W]
        self.learning_rate = learning_rate

        self.x = None
        self.h_prev = 0  # h_prev is the h_0 at the beginning
        self.y_pred = None
        self.h_next = None
        self.loss = None
        self.grad = None
        self.next_grad = 0
        self.y = None

        self.W_hx = None
        self.W_hh = 0.01 * np.random.randn(self.state_size, self.state_size)
        self.W_hy = None
        self.b_h = 0.01 * np.random.randn(self.state_size)
        self.b_0 = 0

    def feedforward(self, x, y=None, weights_and_biases=None):
        self.x = x
        self.y = y

        if self.W_hh is None:
            self.__init_weights()
        elif weights_and_biases:
            [
                self.W_hh,
                self.W_hx,
                self.b_h,
                self.W_hy,
                self.b_0,
            ] = weights_and_biases

        self.h_next = self.f_W(self.W_hh @ self.h_prev + self.W_hx @ self.x + self.b_h)
        self.y_pred = self.W_hy @ self.h_next + self.b_0
        self.loss = mean_squared_error(self.y_pred, self.y)

    def backpropagation(self):
        self.grad = [
            2 * (self.y_pred - self.y) * self.W_hy * self.f_W.derivative(self.x) * self.h_prev,  # W_hh
            2 * (self.y_pred - self.y) * self.W_hy * self.f_W.derivative(self.x) * self.x,  # W_hx
            2 * (self.y_pred - self.y) * self.W_hy * self.f_W.derivative(self.x),  # b_h
            2 * (self.y_pred - self.y) * self.h_next,  # W_hy
            2 * (self.y_pred - self.y),  # b_0
        ]
        self.grad += self.next_grad

    def __init_weights(self):
        x_shape = self.x.shape
        if len(x_shape) == 2:
            x_shape = x_shape[1]

        self.W_hx = 0.01 * np.random.randn(self.state_size, x_shape)
        self.W_hy = 0.01 * np.random.randn(self.state_size, self.y.shape)


class VanillaRNN:
    def __init__(self,  n_iter=1000):
        self.n_iter = n_iter

        self.X = None
        self.y = None
        self.n = None

        self.rnn_layers = []

    def train(self, X, y):
        self.X = np.array(X)
        self.y = np.array(y)
        self.n = self.X.shape[1]

        for i in range(self.n):
            self.rnn_layers.append(RNNBlock("tanh", state_size=8, learning_rate=0.03))

        weights_and_biases = None

        for _ in range(self.n_iter):
            for j in range(self.X.shape[0]):

                self.rnn_layers[0].feedforward(self.X[:, 0, :], self.y[0], weights_and_biases)
                loss = 0
                for i in range(1, self.n):
                    self.rnn_layers[i].h_prev = self.rnn_layers[i - 1].h_next
                    self.rnn_layers[i].feedforward(
                        self.X[:, i, :], self.y[i], weights_and_biases
                    )
                    loss += self.rnn_layers[j].loss

                self.rnn_layers[self.n - 1].backpropagation()
                grad = self.rnn_layers[self.n - 1].grad
                for i in reversed(range(0, self.n - 2)):
                    self.rnn_layers[i].grad_next = self.rnn_layers[j + 1].grad
                    self.rnn_layers[i].backpropagation()
                    grad += self.rnn_layers[i].grad

                weights_and_biases = [
                    self.rnn_layers[0].W_hh,
                    self.rnn_layers[0].W_hx,
                    self.rnn_layers[0].b_h,
                    self.rnn_layers[0].W_hy,
                    self.rnn_layers[0].b_0,
                ]
                weights_and_biases -= 2 * self.rnn_layers[0].learning_rate * grad

        for i in range(len(self.rnn_layers)):
            [
                self.rnn_layers[i].W_hh,
                self.rnn_layers[i].W_hx,
                self.rnn_layers[i].b_h,
                self.rnn_layers[i].W_hy,
                self.rnn_layers[i].b_0,
            ] = weights_and_biases


VRNN = VanillaRNN()
VRNN.train(dracula_X, dracula_y)