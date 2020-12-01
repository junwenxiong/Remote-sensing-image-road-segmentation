import torch
import torch.nn as nn
from .resnet import resnet18, resnet34, resnet50, resnext50_32x4d
import torch.nn.functional as F
from .dyrelu import DyReLUB


class SpatialAttention2d(nn.Module):
    def __init__(self, channel):
        super(SpatialAttention2d, self).__init__()
        self.squeeze = nn.Conv2d(channel, 1, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        z = self.squeeze(x)
        z = self.sigmoid(z)
        return x * z


class GAB(nn.Module):
    def __init__(self, input_dim, reduction=1):
        super(GAB, self).__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(input_dim,
                               input_dim // reduction,
                               kernel_size=1,
                               stride=1)
        self.conv2 = nn.Conv2d(input_dim // reduction,
                               input_dim,
                               kernel_size=1,
                               stride=1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        z = self.global_avgpool(x)
        z = self.relu(self.conv1(z))
        z = self.sigmoid(self.conv2(z))
        return x * z


class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels=512,
        n_filters=256,
        kernel_size=3,
        is_deconv=False,
        dyrelu=False
    ):
        super().__init__()
        if kernel_size == 3:
            conv_padding = 1
        elif kernel_size == 1:
            conv_padding = 0
        # B, C, H, W -> B, C/4, H, W
        self.conv1 = nn.Conv2d(in_channels,
                               in_channels // 4,
                               kernel_size,
                               padding=1,
                               bias=False)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
    
        if dyrelu:
            self.relu1 = DyReLUB(in_channels // 4)
        else:
            self.relu1 = nn.ReLU(inplace=True)

        # B, C/4, H, W -> B, C/4, H, W
        if is_deconv == True:
            self.deconv2 = nn.ConvTranspose2d(in_channels // 4,
                                              in_channels // 4,
                                              3,
                                              stride=2,
                                              padding=1,
                                              output_padding=conv_padding,
                                              bias=False)
        else:
            self.deconv2 = nn.Upsample(scale_factor=2,
                                       mode='bilinear',
                                       align_corners=True)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
     
        if dyrelu:
            self.relu2 = DyReLUB(in_channels // 4)
        else:
            self.relu2 = nn.ReLU(inplace=True)

       
        # B, C/4, H, W -> B, C, H, W
        self.conv3 = nn.Conv2d(in_channels // 4,
                               n_filters,
                               kernel_size,
                               padding=conv_padding,
                               bias=False)
        self.norm3 = nn.BatchNorm2d(n_filters)
       
        if dyrelu:
            self.relu3 = DyReLUB(n_filters)
        else:
            self.relu3 = nn.ReLU(inplace=True)


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


# add spatial attention and gobal attention module
class Decoderv2(nn.Module):
    def __init__(self, in_channels, n_filters, kernel_size=3, dyrely=False):
        super(Decoderv2, self).__init__()

        self.conv1 = nn.Conv2d(in_channels,
                               n_filters,
                               kernel_size,
                               padding=1,
                               bias=False)
        self.norm1 = nn.BatchNorm2d(n_filters)
        self.relu = nn.ReLU(inplace=True)

        self.deconv2 = nn.Upsample(scale_factor=2,
                                   mode='bilinear',
                                   align_corners=True)

        self.s_att = SpatialAttention2d(n_filters)
        self.c_att = GAB(n_filters, 16)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.deconv2(x)
        s = self.s_att(x)
        c = self.c_att(x)
        return s + c


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes,
                              out_planes,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              bias=False)  # verify bias false
        self.bn = nn.BatchNorm2d(
            out_planes,
            eps=0.001,  # value found in tensorflow
            momentum=0.1,  # default pytorch value
            affine=True)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ResNet18Unet(nn.Module):
    def __init__(
        self,
        num_classes=1,
        num_channels=3,
        is_deconv=False,
        decoder_kernel_size=3,
        pretrained=True,
    ):
        super().__init__()

        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)
        self.pretrained = pretrained

        filters = [64, 128, 256, 512]
        resnet = resnet18(pretrained=self.pretrained)
        self.base_size = 512
        self.crop_size = 512
        self._up_kwargs = {'mode': 'bilinear', 'align_corners': True}
        # self.firstconv = resnet.conv1
        # assert num_channels == 3, "num channels not used now. to use changle first conv layer to support num channels other then 3"
        # try to use 8-channels as first input
        if num_channels == 3:
            self.firstconv = resnet.conv1
        else:
            self.firstconv = nn.Conv2d(num_channels,
                                       64,
                                       kernel_size=(7, 7),
                                       stride=(2, 2),
                                       padding=(3, 3),
                                       bias=False)

        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        # Decoder
        self.center = DecoderBlock(in_channels=filters[3],
                                   n_filters=filters[3],
                                   kernel_size=decoder_kernel_size,
                                   is_deconv=is_deconv)
        self.decoder4 = DecoderBlock(in_channels=filters[3] + filters[2],
                                     n_filters=filters[2],
                                     kernel_size=decoder_kernel_size,
                                     is_deconv=is_deconv)
        self.decoder3 = DecoderBlock(in_channels=filters[2] + filters[1],
                                     n_filters=filters[1],
                                     kernel_size=decoder_kernel_size,
                                     is_deconv=is_deconv)
        self.decoder2 = DecoderBlock(in_channels=filters[1] + filters[0],
                                     n_filters=filters[0],
                                     kernel_size=decoder_kernel_size,
                                     is_deconv=is_deconv)
        self.decoder1 = DecoderBlock(in_channels=filters[0] + filters[0],
                                     n_filters=filters[0],
                                     kernel_size=decoder_kernel_size,
                                     is_deconv=is_deconv)

        self.finalconv = nn.Sequential(
            nn.Conv2d(filters[0], 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(), nn.Dropout2d(0.1, False),
            nn.Conv2d(32, num_classes, 1))

    def require_encoder_grad(self, requires_grad):
        blocks = [
            self.firstconv, self.encoder1, self.encoder2, self.encoder3,
            self.encoder4
        ]

        for block in blocks:
            for p in block.parameters():
                p.requires_grad = requires_grad

    def forward(self, x):
        # stem
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x_ = self.firstmaxpool(x)

        # Encoder
        e1 = self.encoder1(x_)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        center = self.center(e4)

        d4 = self.decoder4(torch.cat([center, e3], 1))
        d3 = self.decoder3(torch.cat([d4, e2], 1))
        d2 = self.decoder2(torch.cat([d3, e1], 1))
        d1 = self.decoder1(torch.cat([d2, x], 1))

        f = self.finalconv(d1)
        return F.sigmoid(f)


class ResNet34Unet(nn.Module):
    def __init__(
        self,
        num_classes=1,
        num_channels=3,
        is_deconv=False,
        decoder_kernel_size=3,
        pretrained=True,
        dyrelu=False,
    ):
        super().__init__()

        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)
        self.pretrained = pretrained
        filters = [64, 128, 256, 512]
        resnet = resnet34(pretrained=self.pretrained)
        self.base_size = 512
        self.crop_size = 512
        self._up_kwargs = {'mode': 'bilinear', 'align_corners': True}
        # self.firstconv = resnet.conv1
        # assert num_channels == 3, "num channels not used now. to use changle first conv layer to support num channels other then 3"
        # try to use 8-channels as first input
        if num_channels == 3:
            self.firstconv = resnet.conv1
        else:
            self.firstconv = nn.Conv2d(num_channels,
                                       64,
                                       kernel_size=(7, 7),
                                       stride=(2, 2),
                                       padding=(3, 3),
                                       bias=False)

        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        # Decoder
        self.center = DecoderBlock(in_channels=filters[3],
                                   n_filters=filters[3],
                                   kernel_size=decoder_kernel_size,
                                   is_deconv=is_deconv,
                                   dyrelu=dyrelu)
        self.decoder4 = DecoderBlock(in_channels=filters[3] + filters[2],
                                     n_filters=filters[2],
                                     kernel_size=decoder_kernel_size,
                                     is_deconv=is_deconv,
                                     dyrelu=dyrelu)
        self.decoder3 = DecoderBlock(in_channels=filters[2] + filters[1],
                                     n_filters=filters[1],
                                     kernel_size=decoder_kernel_size,
                                     is_deconv=is_deconv,
                                     dyrelu=dyrelu)
        self.decoder2 = DecoderBlock(in_channels=filters[1] + filters[0],
                                     n_filters=filters[0],
                                     kernel_size=decoder_kernel_size,
                                     is_deconv=is_deconv,
                                     dyrelu=dyrelu)
        self.decoder1 = DecoderBlock(in_channels=filters[0] + filters[0],
                                     n_filters=filters[0],
                                     kernel_size=decoder_kernel_size,
                                     is_deconv=is_deconv,
                                     dyrelu=dyrelu)

        self.finalconv1 = nn.Conv2d(filters[0], 32, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32) 

        if dyrelu:
            self.relu = DyReLUB(32)
        else:
            self.relu = nn.ReLU(inplace=True)

        self.finalconv2 = nn.Conv2d(32, num_classes, 1)
        self.sigmoid = nn.Sigmoid()

        self.finalconv = nn.Sequential(
            nn.Conv2d(filters[0], 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(), nn.Dropout2d(0.1, False),
            nn.Conv2d(32, num_classes, 1))
            
    def require_encoder_grad(self, requires_grad):
        blocks = [
            self.firstconv, self.encoder1, self.encoder2, self.encoder3,
            self.encoder4
        ]

        for block in blocks:
            for p in block.parameters():
                p.requires_grad = requires_grad

    def forward(self, x):
        # stem
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x_ = self.firstmaxpool(x)

        # Encoder
        e1 = self.encoder1(x_)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        center = self.center(e4)

        d4 = self.decoder4(torch.cat([center, e3], 1))
        d3 = self.decoder3(torch.cat([d4, e2], 1))
        d2 = self.decoder2(torch.cat([d3, e1], 1))
        d1 = self.decoder1(torch.cat([d2, x], 1))

        # f = self.finalconv(d1)
        f = self.finalconv1(d1)
        f = self.bn1(f)
        f = self.relu(f)
        f = self.finalconv2(f)
        f = self.sigmoid(f)
        return f


class ResNeXt50Unet(nn.Module):
    def __init__(
        self,
        num_classes=1,
        num_channels=3,
        is_deconv=False,
        decoder_kernel_size=3,
        pretrained=True,
    ):
        super().__init__()

        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)
        self.pretrained = pretrained
        filters = [64, 256, 512, 1024, 2048]
        resnet = resnext50_32x4d(pretrained=self.pretrained)
        self.base_size = 512
        self.crop_size = 512
        self._up_kwargs = {'mode': 'bilinear', 'align_corners': True}
        # self.firstconv = resnet.conv1
        # assert num_channels == 3, "num channels not used now. to use changle first conv layer to support num channels other then 3"
        # try to use 8-channels as first input
        if num_channels == 3:
            self.firstconv = resnet.conv1
        else:
            self.firstconv = nn.Conv2d(num_channels,
                                       64,
                                       kernel_size=(7, 7),
                                       stride=(2, 2),
                                       padding=(3, 3),
                                       bias=False)

        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1  # 256
        self.encoder2 = resnet.layer2  # 512
        self.encoder3 = resnet.layer3  # 1024
        self.encoder4 = resnet.layer4  # 2048

        # Decoder
        # increase decoder layer
        # 2048 -> 2048
        self.center = DecoderBlock(in_channels=filters[4],
                                   n_filters=filters[4],
                                   kernel_size=decoder_kernel_size,
                                   is_deconv=is_deconv)
        # 2048 + 1024 -> 1024
        self.decoder4 = DecoderBlock(in_channels=filters[4] + filters[3],
                                     n_filters=filters[3],
                                     kernel_size=decoder_kernel_size,
                                     is_deconv=is_deconv)
        # 1024 + 512 -> 512
        self.decoder3 = DecoderBlock(in_channels=filters[3] + filters[2],
                                     n_filters=filters[2],
                                     kernel_size=decoder_kernel_size,
                                     is_deconv=is_deconv)
        # 512 + 256 -> 256
        self.decoder2 = DecoderBlock(in_channels=filters[2] + filters[1],
                                     n_filters=filters[1],
                                     kernel_size=decoder_kernel_size,
                                     is_deconv=is_deconv)
        # 256 + 64 -> 64
        self.decoder1 = DecoderBlock(in_channels=filters[1] + filters[0],
                                     n_filters=filters[0],
                                     kernel_size=decoder_kernel_size,
                                     is_deconv=is_deconv)

        self.finalconv = nn.Sequential(
            nn.Conv2d(filters[0], 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(), nn.Dropout2d(0.1, False),
            nn.Conv2d(32, num_classes, 1))

    def require_encoder_grad(self, requires_grad):
        blocks = [
            self.firstconv, self.encoder1, self.encoder2, self.encoder3,
            self.encoder4
        ]

        for block in blocks:
            for p in block.parameters():
                p.requires_grad = requires_grad

    def forward(self, x):
        # stem
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x_ = self.firstmaxpool(x)

        # Encoder
        e1 = self.encoder1(x_)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        center = self.center(e4)

        d4 = self.decoder4(torch.cat([center, e3], 1))
        d3 = self.decoder3(torch.cat([d4, e2], 1))
        d2 = self.decoder2(torch.cat([d3, e1], 1))
        d1 = self.decoder1(torch.cat([d2, x], 1))

        f = self.finalconv(d1)
        return F.sigmoid(f)


class ResNeXt50Unetv2(nn.Module):
    def __init__(
        self,
        num_classes=1,
        num_channels=3,
        is_deconv=False,
        decoder_kernel_size=3,
        pretrained=True,
    ):
        super().__init__()

        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)
        self.pretrained = pretrained
        filters = [64, 256, 512, 1024, 2048]
        resnet = resnext50_32x4d(pretrained=self.pretrained)
        self.base_size = 512
        self.crop_size = 512
        self._up_kwargs = {'mode': 'bilinear', 'align_corners': True}
        # self.firstconv = resnet.conv1
        # assert num_channels == 3, "num channels not used now. to use changle first conv layer to support num channels other then 3"
        # try to use 8-channels as first input
        if num_channels == 3:
            self.firstconv = resnet.conv1
        else:
            self.firstconv = nn.Conv2d(num_channels,
                                       64,
                                       kernel_size=(7, 7),
                                       stride=(2, 2),
                                       padding=(3, 3),
                                       bias=False)

        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1  # 256
        self.encoder2 = resnet.layer2  # 512
        self.encoder3 = resnet.layer3  # 1024
        self.encoder4 = resnet.layer4  # 2048

        # Decoder
        # increase decoder layer
        # 2048 -> 2048
        self.center = Decoderv2(
            in_channels=filters[4],
            n_filters=filters[4],
            kernel_size=decoder_kernel_size,
        )
        # 2048 + 1024 -> 1024
        self.decoder4 = Decoderv2(
            in_channels=filters[4] + filters[3],
            n_filters=filters[3],
            kernel_size=decoder_kernel_size,
        )
        # 1024 + 512 -> 512
        self.decoder3 = Decoderv2(
            in_channels=filters[3] + filters[2],
            n_filters=filters[2],
            kernel_size=decoder_kernel_size,
        )
        # 512 + 256 -> 256
        self.decoder2 = Decoderv2(
            in_channels=filters[2] + filters[1],
            n_filters=filters[1],
            kernel_size=decoder_kernel_size,
        )
        # 256 + 64 -> 64
        self.decoder1 = Decoderv2(
            in_channels=filters[1] + filters[0],
            n_filters=filters[0],
            kernel_size=decoder_kernel_size,
        )

        self.finalconv = nn.Sequential(
            nn.Conv2d(filters[0], 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(), nn.Dropout2d(0.1, False),
            nn.Conv2d(32, num_classes, 1))

    def require_encoder_grad(self, requires_grad):
        blocks = [
            self.firstconv, self.encoder1, self.encoder2, self.encoder3,
            self.encoder4
        ]

        for block in blocks:
            for p in block.parameters():
                p.requires_grad = requires_grad

    def forward(self, x):
        # stem
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x_ = self.firstmaxpool(x)

        # Encoder
        e1 = self.encoder1(x_)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        center = self.center(e4)

        d4 = self.decoder4(torch.cat([center, e3], 1))
        d3 = self.decoder3(torch.cat([d4, e2], 1))
        d2 = self.decoder2(torch.cat([d3, e1], 1))
        d1 = self.decoder1(torch.cat([d2, x], 1))

        f = self.finalconv(d1)
        return F.sigmoid(f)