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

    def backward(self, x, y, outputs, hidden_outputs, learning_rate=0.0001):
        # Calculate errors
        errors = outputs - y

        # Calculate gradients for the output neuron
        d_output_weights = np.dot(errors.T, hidden_outputs)
        d_output_bias = np.sum(errors)

        # Update weights and bias of the output neuron
        self.output_neuron.weights -= learning_rate * d_output_weights.flatten()
        self.output_neuron.bias -= learning_rate * d_output_bias

        # Update weights and biases of hidden neurons
        for i, neuron in enumerate(self.hidden_neurons):
            neuron_errors = errors * self.output_neuron.weights[i]
            d_hidden_weights = np.dot(neuron_errors.T, x)
            d_hidden_bias = np.sum(neuron_errors)
            neuron.weights -= learning_rate * d_hidden_weights.flatten()
            neuron.bias -= learning_rate * d_hidden_bias

    def train(self, x, y, epochs=1000, learning_rate=0.0001):
        for epoch in range(epochs):
            total_loss = 0
            all_outputs = []
            all_hidden_outputs = []
            for xi, yi in zip(x, y):
                hidden_outputs = np.array([neuron(xi) for neuron in self.hidden_neurons])
                output = self.output_neuron(hidden_outputs)
                all_outputs.append(output)
                all_hidden_outputs.append(hidden_outputs)
                loss = (output - yi) ** 2
                total_loss += loss

            # Convert lists to arrays
            all_outputs = np.array(all_outputs).reshape(-1, 1)
            all_hidden_outputs = np.array(all_hidden_outputs).reshape(-1, 3)

            # Backward pass and update weights and biases
            self.backward(x, y, all_outputs, all_hidden_outputs, learning_rate)

            # Sum of Square Errors (SSE)
            sse = total_loss.sum()
            if epoch % 100 == 0:
                print(f'Epoch {epoch + 1}/{epochs}, SSE: {sse}')

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
nn.train(x, y, epochs=1000, learning_rate=0.0001)

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
