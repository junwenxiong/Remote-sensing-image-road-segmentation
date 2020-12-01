"""
Codes of LinkNet based on https://github.com/snakers4/spacenet-three
"""
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

nonlinearity = partial(F.relu, inplace=True)


def swish(x):
    return x * F.sigmoid(x)


class Dblock_more_dilate(nn.Module):
    def __init__(self, channel, dyrelu):
        super(Dblock_more_dilate, self).__init__()
        self.dilate1 = nn.Conv2d(channel,
                                 channel,
                                 kernel_size=3,
                                 dilation=1,
                                 padding=1)
        self.dilate2 = nn.Conv2d(channel,
                                 channel,
                                 kernel_size=3,
                                 dilation=2,
                                 padding=2)
        self.dilate3 = nn.Conv2d(channel,
                                 channel,
                                 kernel_size=3,
                                 dilation=4,
                                 padding=4)
        self.dilate4 = nn.Conv2d(channel,
                                 channel,
                                 kernel_size=3,
                                 dilation=8,
                                 padding=8)
        self.dilate5 = nn.Conv2d(channel,
                                 channel,
                                 kernel_size=3,
                                 dilation=16,
                                 padding=16)
        if dyrelu:
            self.relu = DyReLUB(channel)
        else:
            self.relu = nonlinearity

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        dilate1_out = self.relu(self.dilate1(x))
        dilate2_out = self.relu(self.dilate2(dilate1_out))
        dilate3_out = self.relu(self.dilate3(dilate2_out))
        dilate4_out = self.relu(self.dilate4(dilate3_out))
        dilate5_out = self.relu(self.dilate5(dilate4_out))
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out + dilate5_out
        return out


class Dblock(nn.Module):
    def __init__(self, channel, dyrelu):
        super(Dblock, self).__init__()
        self.dilate1 = nn.Conv2d(channel,
                                 channel,
                                 kernel_size=3,
                                 dilation=1,
                                 padding=1)
        self.dilate2 = nn.Conv2d(channel,
                                 channel,
                                 kernel_size=3,
                                 dilation=2,
                                 padding=2)
        self.dilate3 = nn.Conv2d(channel,
                                 channel,
                                 kernel_size=3,
                                 dilation=4,
                                 padding=4)
        self.dilate4 = nn.Conv2d(channel,
                                 channel,
                                 kernel_size=3,
                                 dilation=8,
                                 padding=8)
        if dyrelu:
            self.relu = DyReLUB(channel)
        else:
            self.relu = nonlinearity

        #self.dilate5 = nn.Conv2d(channel, channel, kernel_size=3, dilation=16, padding=16)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        dilate1_out = self.relu(self.dilate1(x))
        dilate2_out = self.relu(self.dilate2(dilate1_out))
        dilate3_out = self.relu(self.dilate3(dilate2_out))
        dilate4_out = self.relu(self.dilate4(dilate3_out))
        #dilate5_out = nonlinearity(self.dilate5(dilate4_out))
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out  # + dilate5_out
        return out


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters, dyrelu, is_deconv=False):
        super(DecoderBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)

        if dyrelu:
            self.relu1 = DyReLUB(in_channels // 4)
        else:
            self.relu1 = nonlinearity

        # if is_deconv == True:
        self.deconv2 = nn.ConvTranspose2d(in_channels // 4,
                                          in_channels // 4,
                                          3,
                                          stride=2,
                                          padding=1,
                                          output_padding=1)
        # else:
        #     self.deconv2 = nn.Upsample(
        #         scale_factor=2,
        #         mode='bilinear',
        #         align_corners=True,
        #     )

        self.norm2 = nn.BatchNorm2d(in_channels // 4)

        if dyrelu:
            self.relu2 = DyReLUB(in_channels // 4)
        else:
            self.relu2 = nonlinearity

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)

        if dyrelu:
            self.relu3 = DyReLUB(in_channels // 4)
        else:
            self.relu3 = nonlinearity

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x


class DinkNet34_less_pool(nn.Module):
    def __init__(self, num_classes=1, dyrelu=False):
        super(DinkNet34_less_pool, self).__init__()

        filters = [64, 128, 256, 512]
        resnet = resnet34(pretrained=True)

        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3

        self.dblock = Dblock_more_dilate(256, dyrelu)

        self.decoder3 = DecoderBlock(filters[2], filters[1], dyrelu)
        self.decoder2 = DecoderBlock(filters[1], filters[0], dyrelu)
        self.decoder1 = DecoderBlock(filters[0], filters[0], dyrelu)

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        if dyrelu:
            self.finalrelu1 = DyReLUB(32)
        else:
            self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        if dyrelu:
            self.finalrelu2 = DyReLUB(32)
        else:
            self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)

        #Center
        e3 = self.dblock(e3)

        # Decoder
        d3 = self.decoder3(e3) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        # Final Classification
        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return F.sigmoid(out)


class DinkNet34(nn.Module):
    def __init__(self,
                 num_classes=1,
                 num_channels=3,
                 pretrained=True,
                 dyrelu=False):
        super(DinkNet34, self).__init__()

        filters = [64, 128, 256, 512]
        resnet = resnet34(pretrained=pretrained)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.dblock = Dblock(512, dyrelu)

        self.decoder4 = DecoderBlock(filters[3], filters[2], False)
        self.decoder3 = DecoderBlock(filters[2], filters[1], False)
        self.decoder2 = DecoderBlock(filters[1], filters[0], False)
        self.decoder1 = DecoderBlock(filters[0], filters[0], False)

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        if dyrelu:
            self.finalrelu1 = DyReLUB(32)
        else:
            self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        if dyrelu:
            self.finalrelu2 = DyReLUB(32)
        else:
            self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, x):
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
        e4 = self.dblock(e4)

        # Decoder
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return F.sigmoid(out)


class DinkNet50(nn.Module):
    def __init__(self, num_classes=1, pretrained=True, dyrelu=False):
        super(DinkNet50, self).__init__()

        filters = [256, 512, 1024, 2048]
        resnet = resnet50(pretrained=pretrained)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.dblock = Dblock_more_dilate(2048, dyrelu)

        self.decoder4 = DecoderBlock(filters[3], filters[2], dyrelu)
        self.decoder3 = DecoderBlock(filters[2], filters[1], dyrelu)
        self.decoder2 = DecoderBlock(filters[1], filters[0], dyrelu)
        self.decoder1 = DecoderBlock(filters[0], filters[0], dyrelu)

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        if dyrelu:
            self.finalrelu1 = DyReLUB(32)
        else:
            self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        if dyrelu:
            self.finalrelu2 = DyReLUB(32)
        else:
            self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, x):
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
        e4 = self.dblock(e4)

        # Decoder
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)
        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return F.sigmoid(out)

# resnext50
class DinkNet50V2(nn.Module):
    def __init__(self, num_classes=1, pretrained=True, dyrelu=False):
        super(DinkNet50V2, self).__init__()

        filters = [256, 512, 1024, 2048]
        resnet = resnext50_32x4d(pretrained=pretrained)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.dblock = Dblock_more_dilate(2048, dyrelu)

        self.decoder4 = DecoderBlock(filters[3], filters[2], dyrelu)
        self.decoder3 = DecoderBlock(filters[2], filters[1], dyrelu)
        self.decoder2 = DecoderBlock(filters[1], filters[0], dyrelu)
        self.decoder1 = DecoderBlock(filters[0], filters[0], dyrelu)

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        if dyrelu:
            self.finalrelu1 = DyReLUB(32)
        else:
            self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        if dyrelu:
            self.finalrelu2 = DyReLUB(32)
        else:
            self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, x):
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
        e4 = self.dblock(e4)

        # Decoder
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)
        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return F.sigmoid(out)


# resnest50
class DinkNet50V3(nn.Module):
    def __init__(self, num_classes=1, pretrained=True, dyrelu=False):
        super(DinkNet50V3, self).__init__()

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

        self.dblock = Dblock_more_dilate(2048, dyrelu)

        self.decoder4 = DecoderBlock(filters[3], filters[2], dyrelu)
        self.decoder3 = DecoderBlock(filters[2], filters[1], dyrelu)
        self.decoder2 = DecoderBlock(filters[1], filters[0], dyrelu)
        self.decoder1 = DecoderBlock(filters[0], filters[0], dyrelu)

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        if dyrelu:
            self.finalrelu1 = DyReLUB(32)
        else:
            self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        if dyrelu:
            self.finalrelu2 = DyReLUB(32)
        else:
            self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, x):
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
        e4 = self.dblock(e4)

        # Decoder
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)
        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return F.sigmoid(out)


class DinkNet34_FPN4(nn.Module):
    def __init__(self, num_classes=1, num_channels=3):
        super(DinkNet34_FPN4, self).__init__()

        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=False)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.dblock = Dblock(512, False)

        self.decoder4 = DecoderBlock(filters[3], filters[2], False)
        self.decoder3 = DecoderBlock(filters[2], filters[1], False)
        self.decoder2 = DecoderBlock(filters[1], filters[0], False)
        self.decoder1 = DecoderBlock(filters[0], filters[0], False)

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0] * 8, 32, 4, 2, 1)
        self.finalrelu1 = torch.nn.ELU(True)

        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = torch.nn.ELU(True)
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)  # 256
        e1 = self.encoder1(x)  # 256
        e2 = self.encoder2(e1)  # 128
        e3 = self.encoder3(e2)  # 64
        e4 = self.encoder4(e3)  # 32

        # Center
        e4 = self.dblock(e4)  # 32

        # Decoder
        d4 = self.decoder4(e4) + e3  # 256  64
        d3 = self.decoder3(d4) + e2  # 64  128
        d2 = self.decoder2(d3) + e1  # 64  256
        d1 = self.decoder1(d2)  #64  512

        f = torch.cat(
            (d1,
             F.upsample(
                 d2, scale_factor=2, mode='bilinear', align_corners=True),
             F.upsample(
                 d3, scale_factor=4, mode='bilinear', align_corners=True),
             F.upsample(
                 d4, scale_factor=8, mode='bilinear', align_corners=True)),
            1)  #

        out = self.finaldeconv1(f)  #32 1024
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return F.sigmoid(out)


class DinkNet101(nn.Module):
    def __init__(self, num_classes=1, dyrelu=False):
        super(DinkNet101, self).__init__()

        filters = [256, 512, 1024, 2048]
        resnet = resnet101(pretrained=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.dblock = Dblock_more_dilate(2048, dyrelu)

        self.decoder4 = DecoderBlock(filters[3], filters[2], dyrelu)
        self.decoder3 = DecoderBlock(filters[2], filters[1], dyrelu)
        self.decoder2 = DecoderBlock(filters[1], filters[0], dyrelu)
        self.decoder1 = DecoderBlock(filters[0], filters[0], dyrelu)

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        if dyrelu:
            self.finalrelu1 = DyReLUB(32)
        else:
            self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        if dyrelu:
            self.finalrelu2 = DyReLUB(32)
        else:
            self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, x):
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
        e4 = self.dblock(e4)

        # Decoder
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)
        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return F.sigmoid(out)


class LinkNet34(nn.Module):
    def __init__(self, num_classes=1):
        super(LinkNet34, self).__init__()

        filters = [64, 128, 256, 512]
        resnet = resnet34(pretrained=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 3, stride=2)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 2, padding=1)

    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Decoder
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)
        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return F.sigmoid(out)