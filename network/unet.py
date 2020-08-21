""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, mid_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class DoubleFull(nn.Module):
    """ Fully connected layers, flatten >> FC >> Relu >> Fc >> (Relu) >> (unflatten)"""
    def __init__(self, in_sizes, med_sizes, out_sizes, last_layer_activated=True, unflatten=True):
        super().__init__()
        self.unflatten = unflatten
        self.kirkhar = nn.Flatten()
        modules = [nn.Linear(in_sizes, med_sizes),
                   nn.ReLU(inplace=True),
                   nn.Linear(med_sizes, out_sizes)
                   ]
        if last_layer_activated:
            modules.append(nn.ReLU(inplace=True))
        self.double_full = nn.Sequential(*modules)

    def forward(self, x):
        x1 = self.kirkhar(x)
        x2 = self.double_full(x1)
        if self.unflatten:
            return x2.view([x.shape[0], -1, x.shape[2], x.shape[3]])
        else:
            return x2


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, in_channels // 2, out_channels)

    def forward(self, x1, x2):

        x1 = self.up(x1)
        # input is CHW
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super().__init__()

        self.inc = DoubleConv(n_channels, 64, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256)
        self.up2 = Up(512, 128)
        self.up3 = Up(256, 64)
        self.up4 = Up(128, 64)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x

    def num_params(self):
        return


class UNetFC(nn.Module):
    def __init__(self, n_channels):
        super().__init__()

        self.inc = DoubleConv(n_channels, 64, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256)
        self.up2 = Up(512, 128)
        self.up3 = Up(256, 64)
        self.up4 = Up(128, 1)
        self.fc1 = DoubleFull(2500*1, 1000, 2500, last_layer_activated=False)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.fc1(x)
        return x


class Net(nn.Module):
    def __init__(self, n_channels, n_classes):
        super().__init__()
        self.inc1 = DoubleConv(n_channels, 32, 64)
        self.inc2 = DoubleConv(64, 128, 256)
        # self.inc3 = DoubleConv(256, 512, 1024)
        # self.inc4 = DoubleConv(1024, 512, 256)
        self.inc5 = DoubleConv(256, 128, 64)
        self.inc6 = DoubleConv(64, 32, 16)
        self.outc = OutConv(16, n_classes)

    def forward(self, x):
        x1 = self.inc1(x)
        x2 = self.inc2(x1)
        # x4 = self.inc3(x3)
        # x5 = self.inc4(x4)
        x5 = x2
        x6 = self.inc5(x5)
        x7 = self.inc6(x6)
        x8 = self.outc(x7)
        return x8


def get_network(network_id):
    if network_id == 1:
        return UNet(n_channels=1, n_classes=1)
    elif network_id == 2:
        return UNetFC(n_channels=1)
    elif network_id == 3:
        return Net(n_channels=1, n_classes=1)
