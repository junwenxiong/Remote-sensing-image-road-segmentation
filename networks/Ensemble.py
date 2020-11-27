import torch
import torch.nn as nn
import torch.nn.functional as F

class MyEnsemble(nn.Module):
    def __init__(self, in_channels, n_classes=1):
        super(MyEnsemble, self).__init__()
   
        self.conv1 = nn.Conv2d(in_channels, n_classes * 3, 3, 1, 1)
        self.conv2 = nn.Conv2d(n_classes * 3, n_classes, 3, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):

        x = self.conv1(input)
        x = self.conv2(x)
        x = self.sigmoid(x)
        return x
