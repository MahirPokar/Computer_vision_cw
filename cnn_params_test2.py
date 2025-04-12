import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import itertools

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)

# --- Data loading ---
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
train_subset = Subset(trainset, range(10000))  # Smaller subset for speed

# --- Flexible CNN model definition ---
class TunableCNN(nn.Module):
    def __init__(self, dropout=0.3, activation_fn=nn.ReLU, use_batchnorm=False):
        super(TunableCNN, self).__init__()
        def conv_block(in_ch, out_ch):
            layers = [nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)]
            if use_batchnorm:
                layers.append(nn.BatchNorm2d(out_ch))
            layers.append(activation_fn())
            layers.append(nn.MaxPool2d(2))
            return nn.Sequential(*layers)

        self.conv = nn.Sequential(
            conv_block(3, 32),
            conv_block(32, 64),
            conv_block(64, 128)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.Dropout(dropout),
            activation_fn(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        return self.fc(self.conv(x))

# --- Training function ---
def train_model(config, epochs=10):
    model = TunableCNN(
        dropout=config['dropout'],
        activation_fn=config['activation'],
        use_batchnorm=config['batchnorm']
    )

    trainloader = DataLoader(train_subset, batch_size=config['batch_size'], shuffle=True)
    testloader = DataLoader(testset, batch_size=100, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = config['optimizer'](model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])

    history = {'train_acc': [], 'test_acc': []}

    for epoch in range(epochs):
        # Training
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
        history['train_acc'].append(100. * correct / total)

        # Evaluation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in testloader:
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        history['test_acc'].append(100. * correct / total)

    return history

# --- Hyperparameter space ---
param_grid = {
    'lr': [0.001, 0.0005],
    'batch_size': [64, 128],
    'dropout': [0.3, 0.5],
    'activation': [nn.ReLU, nn.LeakyReLU],
    'batchnorm': [False, True],
    'weight_decay': [0.0, 1e-4],
    'optimizer': [optim.Adam, optim.SGD]
}

# Create combinations
param_names = list(param_grid.keys())
param_combos = list(itertools.product(*[param_grid[name] for name in param_names]))

# --- Run training for each config ---
results = {}
for i, combo in enumerate(param_combos):
    config = dict(zip(param_names, combo))
    key = f"config_{i+1}"
    print(f"Training {key}: {config}")
    results[key] = {
        'config': config,
        'history': train_model(config)
    }

# --- Save summary CSV ---
summary = []
for key, res in results.items():
    config = res['config']
    final_acc = res['history']['test_acc'][-1]
    summary.append({**config, 'final_test_acc': final_acc})

summary_df = pd.DataFrame(summary)
summary_df.to_csv("hparam_tuning_summary.csv", index=False)

# --- Plot top 5 configs ---
top5 = summary_df.sort_values(by='final_test_acc', ascending=False).head(5)
plt.figure(figsize=(10, 6))
for index in top5.index:
    key = f"config_{index + 1}"
    acc = results[key]['history']['test_acc']
    label = f"{key}: {top5.loc[index, 'final_test_acc']:.2f}%"
    plt.plot(acc, label=label)

plt.title("Top 5 Configs - Test Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.tight_layout()
plt.savefig("hparam_top5_test_accuracy.png", dpi=300)
plt.show()
