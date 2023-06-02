"""
     You need to do the same implementation of the previous homework but with tensorflow. However, you cannot use dense
     layer or dense network implementations of tf or keras. Here are some notes
     - you can use all tf basic mathematical operations (matmul, mean or calculations etc.)
     - you can use tf.GradientTape() to calculate gradients of the loss function
     - you can use tf.Module and inherit your dense layer and dense network from it. Therefore, you can use all the
       methods and arguments that the tf.Module includes (e.g, .trainable_variables)
     - you cannot use anything else, that is connected to dense or any other layer implementation from tf
       (e.g., tf.keras.layers.Dense)
"""


import tensorflow as tf
from keras.activations import sigmoid, relu, linear
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import numpy as np

np.random.seed(78)


class TenserflowishLayer(tf.Module):
    def __init__(self, input_size, output_size, activation=None):
        super().__init__()
        self.weights = tf.Variable(tf.random.normal([input_size, output_size]), name="W")
        self.biases = tf.Variable(tf.zeros([output_size]), name="b")
        self.activation = activation

        self.output = None
        self.inputs = None

    def feedforward(self, inputs):
        self.inputs = inputs
        self.output = tf.matmul(inputs, self.weights) + self.biases

        for i in range(self.output.shape[1]):
            self.output[:, i] = self.__set_activation(self.output[:, i])

        return self.output

    def backpropagation(self, grad_output, learning_rate):
        activation_derivative = None
        with tf.GradientTape() as tape:
            self.output

        grad_weights = tf.matmul(self.inputs.T, (grad_output * activation_derivative))
        grad_biases = np.sum(grad_output * activation_derivative, axis=0)

        grad_input = tf.matmul((grad_output * activation_derivative), self.weights.T)

        self.weights -= learning_rate * grad_weights
        self.biases -= learning_rate * grad_biases

        return grad_input

    def __set_activation(self, X):
        if not self.activation:
            return linear(X)
        elif self.activation == "sigmoid":
            return sigmoid(X)
        elif self.activation == "relu":
            return relu(X)
        else:
            raise ValueError(
                f"There is no activation like {self.activation}."
            )


class TenserflowishDNN:
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

# X_train_scaled, X_test_scaled = tf.Variable(X_train_scaled), tf.Variable(X_test_scaled)
# y_train, y_test = tf.Variable(y_train), tf.Variable(y_test)

# Train sklearn's LinearRegression model
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)

# Predict with sklearn's LinearRegression model
y_pred_lr = lr_model.predict(X_test_scaled)

# Train the DenseNetwork implemented from scratch
dense_net = TenserflowishDNN()
dense_net.add_layer(TenserflowishLayer(10, 30, activation="relu"))
dense_net.add_layer(TenserflowishLayer(30, 1))

# Train the DenseNetwork using gradient descent
learning_rate = 0.003
num_epochs = 1000
for epoch in range(num_epochs):
    # feedforward pass
    y_pred = tf.Variable(dense_net.feedforward(X_train_scaled))

    # Compute loss (mean squared error)
    if epoch == 400:
        learning_rate /= 10
    print(f"epoch {epoch}:{loss}")
    # backpropagation pass
    with tf.GradientTape() as tape:
        loss = (y_pred - y_train) ** 2

    grad_output = tape.gradient(loss, y_pred) / len(X_train_scaled)
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
