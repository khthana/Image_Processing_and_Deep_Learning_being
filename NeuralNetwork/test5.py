import numpy as np
import matplotlib.pyplot as plt

def generate_points(start=0, end=10, step=0.1):
    # Generate x values from start to end with the given step size
    x_values = np.arange(start, end + step, step)
    num_points = len(x_values)

    # Generate y values as sine of x values
    y_values = np.sin(x_values)

    # Add noise to the y values
    noise = np.random.randn(num_points) * 0.1
    noisy_y_values = y_values + noise

    # Combine x and noisy y values into a list of points
    points = list(zip(x_values, noisy_y_values))

    return points

def relu(x):
    return np.maximum(0, x)

def derivated_relu(x):
    return np.where(x > 0, 1, 0)

class FullyConnectedLayer(object):
    def __init__(self, num_inputs, layer_size, activation_function, derivated_activation_function=None):
        super().__init__()
        self.W = np.random.standard_normal((num_inputs, layer_size))
        self.b = np.random.standard_normal(layer_size)
        self.size = layer_size
        self.activation_function = activation_function
        self.derivated_activation_function = derivated_activation_function
        self.x, self.y = None, None
        self.dL_dW, self.dL_db = None, None

    def forward(self, x):
        z = np.dot(x, self.W) + self.b
        self.y = self.activation_function(z)
        self.x = x
        return self.y

    def backward(self, dL_dy):
        dy_dz = self.derivated_activation_function(self.y)
        dL_dz = dL_dy * dy_dz
        dz_dw = self.x.T
        dz_dx = self.W.T
        dz_db = np.ones(dL_dy.shape[0])

        self.dL_dW = np.dot(dz_dw, dL_dz)
        self.dL_db = np.dot(dz_db, dL_dz)

        dL_dx = np.dot(dL_dz, dz_dx)
        return dL_dx

    def optimize(self, epsilon):
        self.W -= epsilon * self.dL_dW
        self.b -= epsilon * self.dL_db

class Neuron(object):
    def __init__(self, num_inputs, activation_function):
        super().__init__()
        self.W = np.random.uniform(size=num_inputs, low=-1., high=1.)
        self.b = np.random.uniform(size=1, low=-1., high=1.)
        self.activation_function = activation_function

    def forward(self, x):
        z = np.dot(x, self.W) + self.b
        return self.activation_function(z)

def mean_squared_error(pred, target):
    return np.mean(np.square(pred - target))

def derivated_mean_squared_error(pred, target):
    return 2 * (pred - target) / pred.shape[0]

class SimpleNetwork(object):
    def __init__(self, num_inputs, num_outputs, hidden_layers_sizes=(3,),
                activation_function=relu, derivated_activation_function=derivated_relu,
                loss_function=mean_squared_error, derivated_loss_function=derivated_mean_squared_error):
        super().__init__()
        layer_sizes = [num_inputs, *hidden_layers_sizes, num_outputs]
        self.layers = [
            FullyConnectedLayer(layer_sizes[i], layer_sizes[i + 1], activation_function, derivated_activation_function)
            for i in range(len(layer_sizes) - 1)
        ]

        self.loss_function = loss_function
        self.derivated_loss_function = derivated_loss_function

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def predict(self, x):
        estimations = self.forward(x)
        return estimations

    def backward(self, dL_dy):
        for layer in reversed(self.layers):
            dL_dy = layer.backward(dL_dy)
        return dL_dy

    def optimize(self, epsilon):
        for layer in self.layers:
            layer.optimize(epsilon)

    def train(self, X_train, y_train, num_epochs=5, learning_rate=0.001):
        losses = []
        for i in range(num_epochs):
            # Training without mini-batches
            predictions = self.forward(X_train)
            L = self.loss_function(predictions, y_train)
            dL_dy = self.derivated_loss_function(predictions, y_train)
            self.backward(dL_dy)
            self.optimize(learning_rate)
            losses.append(L)
            print(f"Epoch {i+1}/{num_epochs}: training loss = {L:.6f}")
        return losses

def plot_results(network, X_train, y_train):
    predictions = network.predict(X_train)
    plt.scatter(X_train, y_train, label='True Data')
    plt.plot(X_train, predictions, color='red', label='Predictions')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('True Data vs Predictions')
    plt.legend()
    plt.show()

# Generate data points
points = generate_points()
X_train = np.array([p[0] for p in points]).reshape(-1, 1)  # Reshape to (n_samples, n_features)
y_train = np.array([p[1] for p in points]).reshape(-1, 1)

# Initialize the network
network = SimpleNetwork(num_inputs=1, num_outputs=1, hidden_layers_sizes=(8))

# Train the network
network.train(X_train, y_train, num_epochs=1000, learning_rate=0.001)

# Plot the results
plot_results(network, X_train, y_train)
