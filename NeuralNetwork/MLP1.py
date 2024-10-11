import numpy as np
import matplotlib.pyplot as plt

class Neuron:
    def __init__(self, input_size):
        # Initialize weights and bias with small random values
        self.weights = np.random.randn(input_size) * 0.01
        self.bias = 0.0

    def forward(self, x):
        # Linear combination
        z = np.dot(self.weights, x) + self.bias
        return z

    def __call__(self, x):
        # Makes the object callable
        return self.forward(x)

class NeuralNetwork:
    def __init__(self):
        # Initialize three neurons for the hidden layer
        self.hidden_neurons = [Neuron(input_size=1) for _ in range(3)]
        # Initialize one neuron for the output layer, which takes input from 3 hidden neurons
        self.output_neuron = Neuron(input_size=3)

    def forward(self, x):
        # Forward pass through hidden layer
        hidden_outputs = np.array([neuron(x) for neuron in self.hidden_neurons])
        # Forward pass through output layer
        output = self.output_neuron(hidden_outputs)
        return output

    def backward(self, x, y, output, learning_rate=0.001):
        # Calculate error
        error = output - y

        # Update weights and biases of the output neuron (linear backward pass)
        hidden_outputs = np.array([neuron(x) for neuron in self.hidden_neurons])
        self.output_neuron.weights -= learning_rate * error * hidden_outputs.flatten()
        self.output_neuron.bias -= learning_rate * error

        # Update weights and biases of hidden neurons
        for i, neuron in enumerate(self.hidden_neurons):
            neuron_error = error * self.output_neuron.weights[i]
            neuron.weights -= learning_rate * neuron_error * x.flatten()
            neuron.bias -= learning_rate * neuron_error

    def train(self, x, y, epochs=1000, learning_rate=0.001):
        for epoch in range(epochs):
            total_loss = 0
            for xi, yi in zip(x, y):
                output = self.forward(xi)
                loss = (output - yi) ** 2
                total_loss += loss
                self.backward(xi, yi, output, learning_rate)
            avg_loss = total_loss / len(x)
            if epoch % 100 == 0:
              print(f'Epoch {epoch + 1}/{epochs}, Loss: {avg_loss[0]}')

# Function to generate some points
def generate_points(num_points=100):
    np.random.seed(0)
    x = np.random.rand(num_points) * 10
    y = 5 * x + 10 + np.random.randn(num_points) * 2  # Line with some noise
    return np.column_stack((x, y))

# Generate data
data = generate_points()
x = data[:, 0].reshape(-1, 1)
y = data[:, 1].reshape(-1, 1)

# Normalize the data
x_mean = np.mean(x)
x_std = np.std(x)
x = (x - x_mean) / x_std

# Create and train the neural network
nn = NeuralNetwork()
nn.train(x, y, epochs=1000, learning_rate=0.001)

# Test the neural network
test_x = np.array([[5.0]])
test_x_normalized = (test_x - x_mean) / x_std
predicted_y = nn.forward(test_x_normalized)
print(f"Predicted value for input {test_x[0][0]} is {predicted_y}")

# Plot results
plt.scatter(x * x_std + x_mean, y, color='blue', label='Data')
predicted_y_all = np.array([nn.forward(xi) for xi in x])
plt.plot(x * x_std + x_mean, predicted_y_all, color='red', label='Fitted line')
plt.legend()
plt.show()