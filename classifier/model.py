import torch
import torch.nn as nn
from torch.autograd import Variable
from .resnext import resnext50_32x4d

class XNet(nn.Module):
    def __init__(self, num_classes=2, predtrained=True):
        super(XNet, self).__init__()
        self.pretrained = predtrained
        resnet = resnext50_32x4d(pretrained=self.pretrained)

        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes)
        # self.softmax = nn.Softmax(num_classes)

    def forward(self, x):
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        # x = self.softmax(x)
        return x

if __name__ == "__main__":
    x = Variable(torch.randn(1, 3, 256, 256))

    model = XNet()

    y = model(x)

    print(y.shape)
