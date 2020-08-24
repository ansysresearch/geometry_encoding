import torch
import torch.nn as nn
from network.network_components import *


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


class UNet2(nn.Module):
    def __init__(self, n_channels, n_classes):
        super().__init__()

        self.inc = DoubleConv(n_channels, 64, 64, SiLU())
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


class UNet3(nn.Module):
    def __init__(self, n_channels, n_classes):
        super().__init__()

        self.inc = DoubleConv(n_channels, 64, 64, Sin())
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
    if network_id == "UNet":
        return UNet(n_channels=1, n_classes=1)
    elif network_id == "UNet2":
        return UNet2(n_channels=1, n_classes=1)
    elif network_id == "UNet3":
        return UNet3(n_channels=1, n_classes=1)
