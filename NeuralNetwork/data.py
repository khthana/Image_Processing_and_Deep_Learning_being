import torch
import torch.nn as nn
import torch.optim as optim
from numpy import *
import matplotlib.pyplot as plt

x = arange(0, 10, 0.1)
x = expand_dims(x, 1)
noise = 2*random.randn(*x.shape)
y = 5*x+10 + noise

data = concatenate((x,y), axis=1)

x = torch.tensor(data[:, 0], dtype=torch.float32).reshape(-1, 1)
y = torch.tensor(data[:, 1], dtype=torch.float32).reshape(-1, 1)

# Step 2: Define a simple linear model
class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(1, 1)
    
    def forward(self, x):
        return self.linear(x)

model = LinearModel()

# Step 3: Define a loss function and an optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

# Step 4: Train the model
num_epochs = 10000
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    
    outputs = model(x)
    loss = criterion(outputs, y)
    
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# After training, print the learned parameters
[w, b] = model.parameters()
print(f'Learned parameters: w = {w.item():.4f}, b = {b.item():.4f}')

# Plot the results
predicted = model(x).detach().numpy()

plt.plot(data[:, 0], data[:, 1], 'ro', label='Original data')
plt.plot(data[:, 0], predicted, label='Fitted line')
plt.legend()
plt.show()