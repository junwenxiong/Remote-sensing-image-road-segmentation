# Attention UNet2D
import torch
import torch.nn as nn
import torch.nn.functional as F


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

class add_attn(nn.Module):
    def __init__(self, x_channels, g_channels=128):
        super(add_attn, self).__init__()
        self.W = nn.Sequential(
            nn.Conv2d(x_channels,
                      x_channels,
                      kernel_size=1,
                      stride=1,
                      padding=0),
            nn.BatchNorm2d(x_channels))
        self.theta = nn.Conv2d(x_channels,
                               x_channels,
                               kernel_size=2,
                               stride=2,
                               padding=0,
                               bias=False)
        self.phi = nn.Conv2d(g_channels,
                             x_channels,
                             kernel_size=1,
                             stride=1,
                             padding=0,
                             bias=True)
        self.psi = nn.Conv2d(x_channels,
                             out_channels=1,
                             kernel_size=1,
                             stride=1,
                             padding=0,
                             bias=True)


    def forward(self, x, g):
        input_size = x.size()
        batch_size = input_size[0]
        assert batch_size == g.size(0)

        theta_x = self.theta(x)
        theta_x_size = theta_x.size()

        phi_g = F.upsample(self.phi(g),
                           size=theta_x_size[2:],
                           mode='bilinear')

        f = F.relu(theta_x + phi_g, inplace=True)

        sigm_psi_f = F.sigmoid(self.psi(f))
        sigm_psi_f = F.upsample(sigm_psi_f, size=input_size[2:], mode='bilinear')
        y = sigm_psi_f.expand_as(x) * x
        W_y = self.W(y)
        return W_y


class unetCat(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(unetCat, self).__init__()
        self.convT = nn.ConvTranspose2d(dim_in, dim_out, kernel_size=4, stride=2, padding=1)

    def forward(self, input_1, input_2):
        output_2 = self.convT(input_2)
        offset = output_2.size()[2] - input_1.size()[2]
        padding = 2 * [offset // 2, offset // 2]
        output_1 = F.pad(input_1, padding)
        y = torch.cat([output_1, output_2], 1)
        return y


class AttentionUNet2D(nn.Module):
    def __init__(self, n_channels=3, n_classes = 1, useBN=True, pretrained=False):
        super(AttentionUNet2D, self).__init__()
        self.n_classes = n_classes
        self.conv1 = self.add_conv_stage(n_channels, 16, useBN=useBN)
        self.conv2 = self.add_conv_stage(16, 32, useBN=useBN)
        self.conv3 = self.add_conv_stage(32, 64, useBN=useBN)
        self.conv4 = self.add_conv_stage(64, 128, useBN=useBN)

        self.center = self.add_conv_stage(128, 256, useBN=useBN)
        self.gating = self.add_conv(256, 128, useBN=useBN)

        self.attn_1 = add_attn(x_channels=128)
        self.attn_2 = add_attn(x_channels=64)
        self.attn_3 = add_attn(x_channels=32)

        self.cat_1 = unetCat(dim_in=256, dim_out=128)
        self.cat_2 = unetCat(dim_in=128, dim_out=64)
        self.cat_3 = unetCat(dim_in=64, dim_out=32)
        self.cat_4 = unetCat(dim_in=32, dim_out=16)

        self.conv4m = self.add_conv_stage(256, 128, useBN=useBN)
        self.conv3m = self.add_conv_stage(128, 64, useBN=useBN)
        self.conv2m = self.add_conv_stage(64, 32, useBN=useBN)
        self.conv1m = self.add_conv_stage(32, 16, useBN=useBN)

        self.final_conv = nn.Sequential(
            nn.Conv2d(16, n_classes, 3, 1, 1), )  # n_classes
        self.max_pool = nn.MaxPool2d(2)
        self.max_pool1 = nn.MaxPool2d(1)

        self.upsample43 = self.upsample(256, 128)
        self.upsample32 = self.upsample(128, 64)
        self.upsample21 = self.upsample(64, 32)
        self.softmax = nn.Softmax(dim=1)

    def add_conv_stage(self,
                       dim_in,
                       dim_out,
                       kernel_size=3,
                       stride=1,
                       padding=1,
                       useBN=False):
        if useBN:
            return nn.Sequential(
                nn.Conv2d(
                    dim_in,
                    dim_out,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=False),
                nn.BatchNorm2d(dim_out),
                #nn.ReLU(inplace=False),
                FReLU(dim_out),
                nn.Conv2d(
                    dim_out,
                    dim_out,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=False),
                nn.BatchNorm2d(dim_out),
                FReLU(dim_out))
                #nn.ReLU(inplace=False))
        else:
            return nn.Sequential(
                nn.Conv2d(
                    dim_in,
                    dim_out,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=False),
                FReLU(dim_out),
                #nn.ReLU(inplace=False),
                nn.Conv2d(
                    dim_out,
                    dim_out,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=False), FReLU(dim_out)
                #nn.ReLU(inplace=False)
            )

    def upsample(self, ch_coarse, ch_fine, useBN=False):
        if useBN:
            return nn.Sequential(
                nn.ConvTranspose2d(ch_coarse, ch_fine, 2, 2, 0, bias=False),
                nn.BatchNorm2d(ch_fine),
                FReLU(ch_fine)
               # nn.ReLU(inplace=False)
            )
        else:
            return nn.Sequential(
                nn.ConvTranspose2d(ch_coarse, ch_fine, 2, 2, 0, bias=False),
                FReLU(ch_fine)
                #nn.ReLU(inplace=False)
            )

    def add_conv(self,
                 dim_in,
                 dim_out,
                 kernel_size=1,
                 stride=1,
                 padding=1,
                 useBN=False):
        if useBN:
            return nn.Sequential(
                nn.Conv2d(
                    dim_in,
                    dim_out,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=False),
                nn.BatchNorm2d(dim_out),
                FReLU(dim_out)
                #nn.ReLU(inplace=False)
            )

    def forward(self, x):
        # from IPython import embed;embed()
        conv1_out = self.conv1(x)  # (304,304,16)
        conv2_out = self.conv2(self.max_pool(conv1_out))  # (152,152,32)
        conv3_out = self.conv3(self.max_pool(conv2_out))  # (76,76,64)
        conv4_out = self.conv4(self.max_pool(conv3_out))  # (38,38,128)

        center_out = self.center(self.max_pool(conv4_out))  # (19,19,256)
        gating_out = self.gating(center_out)  # (19,19,128)

        attn_1_out = self.attn_1(conv4_out, gating_out)  # (38,38,128)
        attn_2_out = self.attn_2(conv3_out, gating_out)  # (76,76,64)
        attn_3_out = self.attn_3(conv2_out, gating_out)  # (152,152,32)

        cat_1_out = self.cat_1(attn_1_out, center_out)  # (38,38,256)
        conv4m_out = self.conv4m(cat_1_out)  # (38,38,128)
        cat_2_out = self.cat_2(attn_2_out, conv4m_out)  # (76,76,128)
        conv3m_out = self.conv3m(cat_2_out)  # (76,76,64)
        cat_3_out = self.cat_3(attn_3_out, conv3m_out)  # (152,152,64)
        conv2m_out = self.conv2m(cat_3_out)  # (152,152,32)
        cat_4_out = self.cat_4(conv1_out, conv2m_out)  # (304,304,32)
        conv1m_out = self.conv1m(cat_4_out)  # (304,304,16)

        conv0_out = self.final_conv(conv1m_out)
        out = F.sigmoid(conv0_out)

        return out

