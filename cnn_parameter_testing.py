import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# ---------- DATA PREPARATION ----------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

trainset_small = torch.utils.data.Subset(trainset, range(10000))  # Subset for faster tuning

# ---------- MODEL DEFINITION ----------
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256), nn.Dropout(0.3), nn.ReLU(),
            nn.Linear(256, 10)
        )
    def forward(self, x): return self.fc_layers(self.conv_layers(x))

# ---------- TRAINING FUNCTION ----------
def train_model(learning_rate, batch_size, epochs=10):
    model = SimpleCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    trainloader = DataLoader(trainset_small, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=100, shuffle=False)

    train_acc, test_acc = [], []

    for epoch in range(epochs):
        model.train()
        correct, total = 0, 0
        for inputs, labels in trainloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        train_acc.append(100 * correct / total)

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in testloader:
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        test_acc.append(100 * correct / total)

    return train_acc, test_acc

# ---------- HYPERPARAMETER TUNING ----------
configs = [
    {'lr': 0.001, 'bs': 64},
    {'lr': 0.0005, 'bs': 64},
    {'lr': 0.001, 'bs': 128},
    {'lr': 0.001, 'bs': 32}
]
# ---------- HYPERPARAMETER TUNING ----------
results = {}
for cfg in configs:
    key = f"lr={cfg['lr']}_bs={cfg['bs']}"
    print(f"Training config: {key}")
    results[key] = train_model(cfg['lr'], cfg['bs'])

# ---------- PLOTTING ----------
fig, axs = plt.subplots(2, 1, figsize=(10, 8))
for key in results:
    axs[0].plot(results[key][0], label=key)  # Training accuracy
    axs[1].plot(results[key][1], label=key)  # Test accuracy
axs[0].set_title('Training Accuracy')
axs[1].set_title('Test Accuracy')
for ax in axs:
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.legend()
plt.tight_layout()
plt.savefig("hyperparameter_tuning_results.png", dpi=300)
plt.show()

# ---------- FINAL ACCURACY SUMMARY ----------
for key in results:
    print(f"{key} → Final Test Accuracy: {results[key][1][-1]:.2f}%")
