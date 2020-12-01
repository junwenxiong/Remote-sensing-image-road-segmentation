import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.functional import interpolate

up_kwargs = {'mode': 'bilinear', 'align_corners': True}


class FCNHead(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FCNHead, self).__init__()
        inter_channels = in_channels // 4

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels), nn.ReLU(), nn.Dropout(0.1, False),
            nn.Conv2d(inter_channels, out_channels, 1))

    def forward(self, x):
        out = self.conv(x)
        return out


def ASPPConv(in_channels, out_channels, atrous_rate, norm_layer):
    block = nn.Sequential(
        nn.Conv2d(in_channels,
                  out_channels,
                  3,
                  padding=atrous_rate,
                  dilation=atrous_rate,
                  bias=False),
        norm_layer(out_channels),
        nn.ReLU(True),
    )
    return block


class AsppPooling(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, up_kwargs):
        super(AsppPooling, self).__init__()
        self._up_kwargs = up_kwargs
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            norm_layer(out_channels), nn.ReLU(True))

    def forward(self, x):
        _, _, h, w = x.size()
        pool = self.gap(x)
        return interpolate(pool, (h, w), **self._up_kwargs)


class ASPP_Module(nn.Module):
    def __init__(self, in_channels, atrous_rates, norm_layer, up_kwargs):
        super(ASPP_Module, self).__init__()
        out_channels = in_channels // 8
        rate1, rate2, rate3 = tuple(atrous_rates)
        self.b0 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(True))
        self.b1 = ASPPConv(in_channels, out_channels, rate1, norm_layer)
        self.b2 = ASPPConv(in_channels, out_channels, rate2, norm_layer)
        self.b3 = ASPPConv(in_channels, out_channels, rate3, norm_layer)
        self.b4 = AsppPooling(in_channels, out_channels, norm_layer, up_kwargs)

        self.project = nn.Sequential(
            nn.Conv2d(5*out_channels, out_channels, 1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(True),
            nn.Dropout2d(0.5, False))

    def forward(self, x):
        feat0 = self.b0(x)
        feat1 = self.b1(x)
        feat2 = self.b2(x)
        feat3 = self.b3(x)
        feat4 = self.b4(x)
        y = torch.cat((feat0, feat1, feat2, feat3, feat4), 1)
        return self.project(y)


class DeepLabV3Head(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 norm_layer,
                 up_kwargs,
                 atrous_rates=[8, 12, 16]):
        super(DeepLabV3Head, self).__init__()
        inter_channels = in_channels // 8
        self.aspp = ASPP_Module(in_channels, atrous_rates, norm_layer, up_kwargs)
        self.block = nn.Sequential(
            nn.Conv2d(inter_channels, inter_channels, 3, padding=1,
                      bias=False), norm_layer(inter_channels), nn.ReLU(True),
            nn.Dropout(0.1, False), nn.Conv2d(inter_channels, out_channels, 1))

    def forward(self, x):
        x = self.aspp(x)
        x = self.block(x)
        return x