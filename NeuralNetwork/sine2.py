# refactor to class Neuron 
import numpy as np
import matplotlib.pyplot as plt

def generate_points(start=0, end=10, step=0.1):
    x_values = np.arange(start, end + step, step)
    num_points = len(x_values)
    y_values = np.sin(x_values)
    noise = np.random.randn(num_points) * 0.1
    noisy_y_values = y_values + noise
    points = list(zip(x_values, noisy_y_values))
    return points

def plot_output(points, model):
    x_values, noisy_y_values = zip(*points)
    x_values = np.array(x_values).reshape(-1, 1)
    noisy_y_values = np.array(noisy_y_values)
    y_values = np.sin(x_values).flatten()
    
    predicted_y_values = model.predict(x_values).flatten()
    
    plt.figure(figsize=(10, 6))
    plt.scatter(x_values, noisy_y_values, label='Noisy Data', color='blue', s=10)
    plt.plot(x_values, y_values, label='True Sine', color='red', linestyle='--')
    plt.plot(x_values, predicted_y_values, label='Model Prediction', color='green')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Model Predictions vs True Sine and Noisy Data')
    plt.legend()
    plt.grid(True)
    plt.show()

class Neuron:
    def __init__(self, input_dim, output_dim, activation=None):
        self.W = np.random.randn(input_dim, output_dim) / np.sqrt(input_dim)
        self.b = np.random.randn(1, output_dim)
        self.activation = activation

    def forward(self, X):
        self.X = X
        self.Z = np.dot(X, self.W) + self.b
        if self.activation:
            self.A = self.activation.forward(self.Z)
        else:
            self.A = self.Z
        return self.A

    def backward(self, dA):
        if self.activation:
            dZ = self.activation.backward(self.Z, dA)
        else:
            dZ = dA
        dW = np.dot(self.X.T, dZ)
        db = np.sum(dZ, axis=0, keepdims=True)
        dX = np.dot(dZ, self.W.T)
        return dW, db, dX

class Tanh:
    def forward(self, X):
        return np.tanh(X)

    def backward(self, X, top_diff):
        output = self.forward(X)
        return (1.0 - np.square(output)) * top_diff

class Model:
    def __init__(self, layers_dim):
        self.layers = []
        input_dim = 1
        for output_dim in layers_dim[:-1]:
            self.layers.append(Neuron(input_dim, output_dim, activation=Tanh()))
            input_dim = output_dim
        self.layers.append(Neuron(input_dim, layers_dim[-1]))  # No activation for the last layer
        self.loss = []

    def calculate_loss(self, X, y):
        mse_loss = np.mean((self.predict(X) - y) ** 2)
        return mse_loss

    def predict(self, X):
        input = X
        for layer in self.layers:
            input = layer.forward(input)
        return input

    def train(self, X, y, num_passes=20000, epsilon=0.01, print_loss=False):
        for epoch in range(num_passes):
            # Forward propagation
            input = X
            for layer in self.layers:
                input = layer.forward(input)

            # Calculate the loss (Mean Squared Error)
            loss = np.mean((input - y) ** 2)
            self.loss.append(loss)

            # Back propagation
            dA = 2 * (input - y) / y.size
            for layer in reversed(self.layers):
                dW, db, dA = layer.backward(dA)
                layer.W -= epsilon * dW
                layer.b -= epsilon * db

            if print_loss and epoch % 1000 == 0:
                print(f"Loss after iteration {epoch}: {loss:.6f}")

        plt.plot(self.loss)
        plt.title('Loss over time')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.show()

# Generate the dataset
points = generate_points()

# Define and train the model
layers_dim = [20, 20, 1]
model = Model(layers_dim)
x = np.array([point[0] for point in points]).reshape(-1, 1)
y = np.array([point[1] for point in points]).reshape(-1, 1)
model.train(x, y, num_passes=20000, epsilon=0.01, print_loss=True)

# Plot the output
plot_output(points, model)
