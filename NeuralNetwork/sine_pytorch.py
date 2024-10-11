import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def generate_points(start=0, end=10, step=0.1):
    x_values = np.arange(start, end + step, step)
    num_points = len(x_values)
    y_values = np.sin(x_values)
    noise = np.random.randn(num_points) * 0.1
    noisy_y_values = y_values + noise
    points = list(zip(x_values, noisy_y_values))
    return points

def plot_output(points, model, loss, device):
    x_values, noisy_y_values = zip(*points)
    x_values = torch.tensor(x_values, dtype=torch.float32).reshape(-1, 1).to(device)
    noisy_y_values = np.array(noisy_y_values)
    y_values = np.sin(x_values.cpu().numpy()).flatten()
    
    model.eval()
    with torch.no_grad():
        predicted_y_values = model(x_values).cpu().numpy().flatten()
    
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.scatter(x_values.cpu().numpy(), noisy_y_values, label='Noisy Data', color='blue', s=10)
    plt.plot(x_values.cpu().numpy(), y_values, label='True Sine', color='red', linestyle='--')
    plt.plot(x_values.cpu().numpy(), predicted_y_values, label='Model Prediction', color='green')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Model Predictions vs True Sine and Noisy Data')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(loss)
    plt.title('Loss over time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

# Check for CUDA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# Generate the dataset
points = generate_points()

# Prepare the data
x = np.array([point[0] for point in points]).reshape(-1, 1)
y = np.array([point[1] for point in points]).reshape(-1, 1)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Convert data to PyTorch tensors
x_train = torch.tensor(x_train, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
x_test = torch.tensor(x_test, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test, dtype=torch.float32).to(device)

# Define model using nn.Sequential
model = nn.Sequential(
    nn.Linear(1, 200),
    nn.ReLU(),
    nn.Linear(200, 200),
    nn.ReLU(),
    nn.Linear(200, 1)
).to(device)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Train the model
num_epochs = 50000
loss_history = []

model.train()
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(x_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    
    loss_history.append(loss.item())
    
    if epoch % 1000 == 0:
        print(f"Loss after iteration {epoch}: {loss.item():.6f}")

# Evaluate the model
model.eval()
with torch.no_grad():
    test_outputs = model(x_test)
    test_loss = criterion(test_outputs, y_test)
    print(f"Test Loss: {test_loss.item():.6f}")

# Plot the output
plot_output(points, model, loss_history, device)
