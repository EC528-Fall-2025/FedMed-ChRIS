import torch
import torch.nn as nn
import torch.nn.functional as F


# Small CNN stub for first sprint tomorrwo
class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)         # 28x28 to 28x28
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)        # 28x28 to 28x28 w 64 channels
        self.pool  = nn.MaxPool2d(2,2)                      # 28x28 to 14x14
        self.drop  = nn.Dropout(0.25)
        self.fc1   = nn.Linear(64*14*14, 128)               # flatten here and dense layers after
        self.fc2   = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.drop(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)
        return x
