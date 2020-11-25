import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F
# import megengine.functional as F
# import megengine.module as M

class FReLU(nn.Module):
    r""" FReLU formulation. The funnel condition has a window size of kxk. (k=3 by default)
    """

    def __init__(self, in_channels):
        super().__init__()
        self.conv_frelu = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels)
        self.bn_frelu = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        y = self.conv_frelu(x)
        y = self.bn_frelu(y)
        x = torch.max(x, y)
        return x

class Google_Conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 1*1 卷积层
        self.google_conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels//4, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_channels//4),
            FReLU(out_channels//4)
        )
        # 1*1  3*3卷积
        self.google_conv2 = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1, padding=0),
            nn.BatchNorm2d(1),
            nn.Conv2d(1, out_channels//2, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels // 2),
            FReLU(out_channels // 2)
        )
        # 1*1 5*5
        self.google_conv3 = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1, padding=0),
            nn.BatchNorm2d(1),
            nn.Conv2d(1, out_channels//8, kernel_size=5, padding=2),
            nn.BatchNorm2d(out_channels // 8),
            FReLU(out_channels//8)
        )
        #
        self.google_conv4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, out_channels//8, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_channels // 8),
            FReLU(out_channels // 8)
        )

    def forward(self, x):
        a = self.google_conv1(x)
        b = self.google_conv2(x)
        c = self.google_conv3(x)
        d = self.google_conv4(x)

        return torch.cat([self.google_conv1(x), self.google_conv2(x), self.google_conv3(x), self.google_conv4(x)],dim =1 )

class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            FReLU(mid_channels),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            FReLU(out_channels)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class G_Down(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            Google_Conv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)



class Up(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels = 3, n_classes = 1):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = DoubleConv(n_channels, 32) #256
        self.down0 = Down(32, 64) #128*128
        self.down1 = Down(64, 128) #64*64
        self.down2 = Down(128, 256) #32*32
        self.down3 = G_Down(256, 512) #16*16
        self.down4 = G_Down(512, 512) #8*8
        self.up1 = Up(1024, 256)
        self.up2 = Up(512, 128)
        self.up3 = Up(256, 64)
        self.up4 = Up(128, 32)
        self.up5 = Up(64, 32)
        self.outc = OutConv(32, n_classes)

    def forward(self, x):
        x0 = self.inc(x) # 32
        x1 = self.down0(x0) # 64
        x2 = self.down1(x1) # 128
        x3 = self.down2(x2) # 256
        x4 = self.down3(x3) # 512
        x5 = self.down4(x4) #512*64*64

        x = self.up1(x5, x4) # output 256
        x = self.up2(x, x3) # output128
        x = self.up3(x, x2) # output 64
        x = self.up4(x, x1) # output 32
        x = self.up5(x, x0)
        logits = self.outc(x)

        return F.sigmoid(logits)