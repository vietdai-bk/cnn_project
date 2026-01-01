import torch.nn as nn
import torch.nn.functional as F

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=2, stride=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 1 * 1, 128)
        self.fc2 = nn.Linear(128, 10)
    def forward(self, x):
        x = self.conv1(x) # [B, 32, 14, 14]
        x = F.relu(x)
        x = self.pool(x) # [B, 32, 7, 7]
        x = self.conv2(x) # [B, 64, 3, 3]
        x = F.relu(x)
        x = self.pool(x) # [B, 64, 1, 1]
        x = x.view(x.size(0), -1) # [B, 64]
        x = self.fc1(x) # [64, 128]
        x = F.relu(x)
        x = self.fc2(x) # [128, 10] => 10 class 0 -> 9
        return x