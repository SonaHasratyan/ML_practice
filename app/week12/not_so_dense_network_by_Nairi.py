from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import numpy as np


class DenseLayer:
    def __init__(self, input_size, output_size, activation="sigmoid"):
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.biases = np.zeros(output_size)

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

        return self.output

    def backward(self, grad_output, learning_rate):
        grad_weights = np.dot(self.inputs.T, grad_output)
        grad_biases = np.sum(grad_output, axis=0)

        grad_input = np.dot(grad_output, self.weights.T)
        self.weights -= learning_rate * grad_weights
        self.biases -= learning_rate * grad_biases

        return grad_input


class DenseNetwork:
    def __init__(self):
        self.layers = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def forward(self, inputs):
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def backward(self, grad_output, learning_rate):
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output, learning_rate)


# Generate synthetic dataset
X, y = make_regression(n_samples=100, n_features=10, noise=0.5, random_state=42)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

# Standardize the input features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train scikit-learn's LinearRegression model
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)

# Predict with scikit-learn's LinearRegression model
y_pred_lr = lr_model.predict(X_test_scaled)

# Train the DenseNetwork implemented from scratch
dense_net = DenseNetwork()
dense_net.add_layer(DenseLayer(10, 10))
dense_net.add_layer(DenseLayer(10, 1))

# Train the DenseNetwork using gradient descent
learning_rate = 0.001
num_epochs = 1000
for epoch in range(num_epochs):
    # Forward pass
    y_pred = dense_net.forward(X_train_scaled)

    # Compute loss (mean squared error)
    loss = np.mean((y_pred - y_train) ** 2)
    print(f'epoch {epoch}:{loss}')
    # Backward pass
    grad_output = 2 * (y_pred - y_train) / len(X_train_scaled)
    dense_net.backward(grad_output, learning_rate)

# Predict with the DenseNetwork
y_pred_dense = dense_net.forward(X_test_scaled)

# Compare the results
print(
    "Mean Squared Error (sklearn LinearRegression):",
    mean_squared_error(y_test, y_pred_lr),
)
print(
    "Mean Squared Error (DenseNetwork implemented from scratch):",
    mean_squared_error(y_test, y_pred_dense),
)
