import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Custom KANLinear layer
class KANLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, grid_size=5, spline_order=3, scale_noise=0.1, 
                 scale_base=1.0, scale_spline=1.0, enable_standalone_scale_spline=True, 
                 base_activation=torch.nn.SiLU, grid_eps=0.02, grid_range=[-1, 1]):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = ((torch.arange(-spline_order, grid_size + spline_order + 1) * h + grid_range[0])
                .expand(in_features, -1).contiguous())
        self.register_buffer("grid", grid)

        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.base_bias = torch.nn.Parameter(torch.Tensor(out_features))
        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline

        # Initialize weights
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        if self.base_bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.base_weight)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.base_bias, -bound, bound)

    def forward(self, input):
        return F.linear(input, self.base_weight, self.base_bias)

# Define a simple neural network with KANLinear layers
class KANNet(nn.Module):
    def __init__(self):
        super(KANNet, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = KANLinear(28 * 28, 128)
        self.fc2 = KANLinear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Hyperparameters
batch_size = 64
learning_rate = 0.001
epochs = 5

# Prepare MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# Initialize model, loss, and optimizer
model = KANNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(epochs):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
    print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}')

# Evaluation on test data
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for data, target in test_loader:
        output = model(data)
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

print(f'Accuracy on test set: {100 * correct / total}%')
