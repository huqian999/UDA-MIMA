from __future__ import print_function, division
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch

from models.layers import conv_block, up_conv


class U_Net_4(nn.Module):

    def __init__(self, in_ch=3, num_classes=7):
        super(U_Net_4, self).__init__()
        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(in_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

        self.Up4 = up_conv(filters[4], filters[3])
        self.Up_conv4 = conv_block(filters[4], filters[3])

        self.Up3 = up_conv(filters[3], filters[2])
        self.Up_conv3 = conv_block(filters[3], filters[2])

        self.Up2 = up_conv(filters[2], filters[1])
        self.Up_conv2 = conv_block(filters[2], filters[1])

        self.Up1 = up_conv(filters[1], filters[0])
        self.Up_conv1 = conv_block(filters[1], filters[0])

        self.cls = nn.Conv2d(filters[0], num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        e1 = self.Conv1(x)

        e2 = self.Maxpool(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool(e4)
        e5 = self.Conv5(e5)

        d4 = self.Up4(e5)
        d4 = torch.cat((e4, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((e3, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((e2, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Up1(d2)
        d1 = torch.cat((e1, d1), dim=1)
        d1 = self.Up_conv1(d1)

        out = self.cls(d1)
        if self.training:
            return d1, out
        else:
            return out


class U_Net_3(nn.Module):

    def __init__(self, in_ch=1, num_classes=7):
        super(U_Net_3, self).__init__()

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(in_ch, 64)
        self.Conv2 = conv_block(64, 128)
        self.Conv3 = conv_block(128, 256)
        self.bottleneck = conv_block(256, 512)

        self.up3 = up_conv(512, 256)
        self.up_conv3 = conv_block(512, 256)

        self.up2 = up_conv(256, 128)
        self.up_conv2 = conv_block(256, 128)

        self.up1 = up_conv(128, 64)
        self.up_conv1 = conv_block(128, 64)

        self.final = nn.Conv2d(64, num_classes, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x):

        e1 = self.Conv1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        bottleneck = self.Maxpool3(e3)
        bottleneck = self.bottleneck(bottleneck)

        d3 = self.up3(bottleneck)
        d3 = torch.cat((e3, d3), dim=1)
        d3 = self.up_conv3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat((e2, d2), dim=1)
        d2 = self.up_conv2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat((e1, d1), dim=1)
        d1 = self.up_conv1(d1)
        output = self.final(d1)
        return output


class conv_block_nested(nn.Module):

    def __init__(self, in_ch, mid_ch, out_ch):
        super(conv_block_nested, self).__init__()
        self.activation = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.bn2(x)
        output = self.activation(x)

        return output


# Nested Unet

class NestedUNet(nn.Module):
    """
    Implementation of this paper:
    https://arxiv.org/pdf/1807.10165.pdf
    """

    def __init__(self, cfg):
        super(NestedUNet, self).__init__()
        in_ch = cfg.in_ch
        num_classes = cfg.num_classes
        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = conv_block_nested(in_ch, filters[0], filters[0])
        self.conv1_0 = conv_block_nested(filters[0], filters[1], filters[1])
        self.conv2_0 = conv_block_nested(filters[1], filters[2], filters[2])
        self.conv3_0 = conv_block_nested(filters[2], filters[3], filters[3])
        self.conv4_0 = conv_block_nested(filters[3], filters[4], filters[4])

        self.conv0_1 = conv_block_nested(filters[0] + filters[1], filters[0], filters[0])
        self.conv1_1 = conv_block_nested(filters[1] + filters[2], filters[1], filters[1])
        self.conv2_1 = conv_block_nested(filters[2] + filters[3], filters[2], filters[2])
        self.conv3_1 = conv_block_nested(filters[3] + filters[4], filters[3], filters[3])

        self.conv0_2 = conv_block_nested(filters[0] * 2 + filters[1], filters[0], filters[0])
        self.conv1_2 = conv_block_nested(filters[1] * 2 + filters[2], filters[1], filters[1])
        self.conv2_2 = conv_block_nested(filters[2] * 2 + filters[3], filters[2], filters[2])

        self.conv0_3 = conv_block_nested(filters[0] * 3 + filters[1], filters[0], filters[0])
        self.conv1_3 = conv_block_nested(filters[1] * 3 + filters[2], filters[1], filters[1])

        self.conv0_4 = conv_block_nested(filters[0] * 4 + filters[1], filters[0], filters[0])

        self.final = nn.Conv2d(filters[0], num_classes, kernel_size=1)

    def forward(self, x):
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.Up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.Up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.Up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.Up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.Up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.Up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.Up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.Up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.Up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.Up(x1_3)], 1))

        output = self.final(x0_4)
        return output

