# Work 
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

class MultiplyNode:
    def forward(self, W, X):
        return np.dot(X, W)

    def backward(self, W, X, dZ):
        dW = np.dot(np.transpose(X), dZ)
        dX = np.dot(dZ, np.transpose(W))
        return dW, dX

class AddNode:
    def forward(self, X, b):
        return X + b

    def backward(self, X, b, dZ):
        dX = dZ * np.ones_like(X)
        db = np.dot(np.ones((1, dZ.shape[0]), dtype=np.float64), dZ)
        return db, dX

class Tanh:
    def forward(self, X):
        return np.tanh(X)

    def backward(self, X, top_diff):
        output = self.forward(X)
        return (1.0 - np.square(output)) * top_diff

class Model:
    def __init__(self, layers_dim):
        self.b = []
        self.W = []
        self.loss = []
        layers_dim = [1, *layers_dim, 1]  # Adjust input and output dimensions for regression
        for i in range(len(layers_dim) - 1):
            self.W.append(np.random.randn(layers_dim[i], layers_dim[i + 1]) / np.sqrt(layers_dim[i]))
            self.b.append(np.random.randn(layers_dim[i + 1]).reshape(1, layers_dim[i + 1]))

    def calculate_loss(self, X, y):
        mse_loss = np.mean((self.predict(X) - y) ** 2)
        return mse_loss

    def predict(self, X):
        mul_node = MultiplyNode()
        add_Node = AddNode()
        layer = Tanh()

        input = X
        for i in range(len(self.W) - 1):
            mul = mul_node.forward(self.W[i], input)
            add = add_Node.forward(mul, self.b[i])
            input = layer.forward(add)

        # For the last layer, we do not apply the Tanh activation
        mul = mul_node.forward(self.W[-1], input)
        add = add_Node.forward(mul, self.b[-1])
        return add

    def train(self, X, y, num_passes=20000, epsilon=0.01, print_loss=False):
        mul_node = MultiplyNode()
        add_node = AddNode()
        layer = Tanh()

        for epoch in range(num_passes):
            # Forward propagation
            input = X
            forward = [(None, None, input)]
            for i in range(len(self.W) - 1):
                mul = mul_node.forward(self.W[i], input)
                add = add_node.forward(mul, self.b[i])
                input = layer.forward(add)
                forward.append((mul, add, input))

            # For the last layer, we do not apply the Tanh activation
            mul = mul_node.forward(self.W[-1], input)
            add = add_node.forward(mul, self.b[-1])
            forward.append((mul, add, add))

            # Calculate the loss (Mean Squared Error)
            loss = np.mean((add - y) ** 2)
            self.loss.append(loss)

            # Back propagation
            dadd = 2 * (forward[-1][2] - y) / y.size
            dtanh = dadd
            for i in range(len(forward) - 1, 0, -1):
                if i != len(forward) - 1:  # Skip Tanh activation for the last layer
                    dadd = layer.backward(forward[i][1], dtanh)
                db, dmul = add_node.backward(forward[i][0], self.b[i - 1], dadd)
                dW, dtanh = mul_node.backward(self.W[i - 1], forward[i - 1][2], dmul)

                self.b[i - 1] -= epsilon * db
                self.W[i - 1] -= epsilon * dW

            if print_loss and epoch % 1000 == 0:
                print("Loss after iteration %i: %f" % (epoch, loss))

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
