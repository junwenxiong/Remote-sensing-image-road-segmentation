import torch
import torch.nn as nn
import torch.nn.functional as F


class MyEnsemble(nn.Module):
    def __init__(self, modelA, modelB, in_channels, n_classes=1):
        super(MyEnsemble, self).__init__()
        self.modelA = modelA
        self.modelB = modelB

        self.modelA.sigmoid = nn.Identity()
        self.modelB.sigmoid = nn.Identity()

        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, n_classes * 3, 3, 1, 1)
        self.conv2 = nn.Conv2d(n_classes * 3, n_classes, 3, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        x1 = self.modelA(input.clone())
        x2 = self.modelB(input)
        x = torch.cat([x1, x2], dim=1)

        x = self.relu(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sigmoid(x)
        return x