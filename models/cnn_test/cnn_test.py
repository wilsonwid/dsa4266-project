import torch
import torch.nn as nn

class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=0
        )

        self.maxpool1 = nn.MaxPool2d(kernel_size=1, stride=2, padding=0)

        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=32,
            kernel_size=3, 
            stride=1, 
            padding=0
        )

        self.maxpool2 = nn.MaxPool2d(kernel_size=1, stride=2, padding=0)

        self.dense1 = nn.Linear(96800, 64)

        self.dropout = nn.Dropout(0.25)

        self.dense2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = x.flatten(start_dim=1)
        x = self.dense1(x)
        x = self.dropout(x)
        x = self.dense2(x)
        return torch.sigmoid(x)