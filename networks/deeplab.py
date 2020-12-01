import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models
import torch.nn.functional as F
from .resnet import resnet34, resnet50, resnet101, resnext50_32x4d
from .pyconvhgresnet import pyconvhgresnet50, pyconvhgresnet101
from functools import partial
from .dyrelu import DyReLUB
from .resnest import resnest50, resnest101
from torch.nn.functional import interpolate
from .customize import FCNHead, DeepLabV3Head

up_kwargs = {'mode': 'bilinear', 'align_corners': True}


class DeepLabV3_FCN(nn.Module):
    def __init__(self, num_classes=1, pretrained=True):
        super(DeepLabV3_FCN, self).__init__()
        self._up_kwargs = up_kwargs
        self.norm_layer = nn.BatchNorm2d
        filters = [256, 512, 1024, 2048]
        resnet = resnest50(pretrained=pretrained)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.head = DeepLabV3Head(filters[3], num_classes, self.norm_layer,
                                  self._up_kwargs)
        self.auxlayer = FCNHead(filters[2], num_classes)

    def forward(self, x):
        _, _, h, w = x.size()
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Center
        outputs = []
        d1 = self.head(e4)
        d1 = interpolate(d1, (h, w), **self._up_kwargs)
        d1 = F.sigmoid(d1)
        outputs.append(d1)
        auxout = self.auxlayer(e3)
        auxout = interpolate(auxout, (h, w), **self._up_kwargs)
        auxout = F.sigmoid(auxout)
        outputs.append(auxout)

        return tuple(outputs)
