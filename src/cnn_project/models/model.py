import torch
import torch.nn as nn
from torchvision import models

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
    def forward(self, x):
        return self.block(x)

class SimpleModel(nn.Module):
    def __init__(self, in_c=1, num_classes=10):
        super(SimpleModel, self).__init__()
        self.block1 = ConvBlock(in_c, 32)
        self.block2 = ConvBlock(32, 64)
        self.dropout = nn.Dropout(p=0.1)
        self.gap = nn.AdaptiveAvgPool2d((3, 3))
        self.fc = nn.Linear(64 * 3 * 3, num_classes)
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x
    
class ShuffleNet(nn.Module):
    def __init__(self, in_c=1, num_classes=10):
        super().__init__()
        self.backbone = models.shufflenet_v2_x0_5(
            weights = models.ShuffleNet_V2_X0_5_Weights.IMAGENET1K_V1
        )
        conv1 = self.backbone.conv1[0]
        new_conv1 = nn.Conv2d(
            in_channels=in_c,
            out_channels=conv1.out_channels,
            kernel_size=conv1.kernel_size,
            stride=conv1.stride,
            padding=conv1.padding,
            bias=False
        )
        new_conv1.weight.data = conv1.weight.data.mean(dim=1, keepdim=True)
        self.backbone.conv1[0] = new_conv1
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
    def forward(self, x):
        return self.backbone(x)