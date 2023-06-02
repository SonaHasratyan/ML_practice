"""
The implementation should include the functionalities needed for the previous week + all layers
should have option to have activation function (None, 'relu' or 'sigmoid'). Feedforward and
backpropagation should be working according to that. The optional task here (mainly for those,
who have done their previous hw with activation functions) is to implement backpropagation not
by gradient descent algorithm, but by Adam.
"""

from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import numpy as np

# todo close v
# np.seterr(all='raise')
np.random.seed(78)


class DenseLayer:
    def __init__(self, input_size, output_size, activation=None):
        self.weights = np.random.randn(input_size, output_size)
        self.biases = np.zeros(output_size)
        self.activation = activation

        self.__activations = {None: Identity(), "sigmoid": Sigmoid(), "relu": ReLu()}

        if self.activation not in self.__activations:
            raise ValueError(
                f"There is no activation like {self.activation}. "
                f"Please provide one of these {self.__activations} or None"
            )

        self.activation = self.__activations[self.activation]

        self.output = None
        self.inputs = None

    def feedforward(self, inputs):
        self.inputs = inputs
        self.output = inputs @ self.weights + self.biases

        for i in range(self.output.shape[1]):
            self.output[:, i] = self.activation(self.output[:, i])

        return self.output

    def backpropagation(self, grad_output, learning_rate):
        activation_derivative = self.activation.derivative(self.output)
        grad_weights = self.inputs.T @ (grad_output * activation_derivative)
        grad_biases = np.sum(grad_output * activation_derivative, axis=0)

        grad_input = (grad_output * activation_derivative) @ self.weights.T

        self.weights -= learning_rate * grad_weights
        self.biases -= learning_rate * grad_biases

        return grad_input


class MoreDenseNetwork:
    def __init__(self):
        self.layers = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def feedforward(self, inputs):
        for layer in self.layers:
            inputs = layer.feedforward(inputs)
        return inputs

    def backpropagation(self, grad_output, learning_rate):
        for layer in reversed(self.layers):
            grad_output = layer.backpropagation(grad_output, learning_rate)


class Activation:
    def __call__(self, X):
        return (X >= 0) * X

    def derivative(self, X):
        return X >= 0


class Sigmoid(Activation):
    def __call__(self, X):
        return 1 / (1 + np.exp(-X, dtype=np.float128))

    def derivative(self, X):
        return X * (1 - X)


class ReLu(Activation):
    def __call__(self, X):
        pass

    def derivative(self, X):
        pass


class Identity(Activation):
    def __call__(self, X):
        return X

    def derivative(self, X):
        return 1


# Generate synthetic dataset
X, y = make_regression(n_samples=100, n_features=10, noise=0.5, random_state=78)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=78
)

y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

# Standardize the input features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train sklearn's LinearRegression model
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)

# Predict with sklearn's LinearRegression model
y_pred_lr = lr_model.predict(X_test_scaled)

# Train the DenseNetwork implemented from scratch
dense_net = MoreDenseNetwork()
dense_net.add_layer(DenseLayer(10, 10, activation="sigmoid"))
dense_net.add_layer(DenseLayer(10, 1))

# Train the DenseNetwork using gradient descent
learning_rate = 0.01
num_epochs = 1000
for epoch in range(num_epochs):
    # feedforward pass
    y_pred = dense_net.feedforward(X_train_scaled)

    # Compute loss (mean squared error)
    loss = mean_squared_error(y_train, y_pred)
    # if epoch == 50:
    #     learning_rate /= 10
    print(f"epoch {epoch}:{loss}")
    # backpropagation pass
    grad_output = 2 * (y_pred - y_train) / len(X_train_scaled)
    dense_net.backpropagation(grad_output, learning_rate)

# Predict with the DenseNetwork
y_pred_dense = dense_net.feedforward(X_test_scaled)

# Compare the results
print(
    "Mean Squared Error (sklearn LinearRegression):",
    mean_squared_error(y_test, y_pred_lr),
)
print(
    "Mean Squared Error (DenseNetwork implemented from scratch):",
    mean_squared_error(y_test, y_pred_dense),
)
