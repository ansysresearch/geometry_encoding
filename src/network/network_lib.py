from math import log2
from src.network.network_components import *

# all networks have a U-Net structure.
# the original template can be found in https://github.com/milesial/Pytorch-UNet/tree/master/unet

# UNet1 has a code with same size of input
# UNet2 has a code with half of the size of input
# UNet3 has a code with quarter of the size of input.


class CNN1(nn.Module):
    def __init__(self, n_channels, n_classes):
        super().__init__()

        self.inconv1 = DoubleConv(n_channels, 128, 128)
        self.inconv2 = DoubleConv(128, 128, 128)
        self.outconv = OutConv(128, n_classes)

    def forward(self, x):
        x1 = self.inconv1(x)
        x2 = self.inconv2(x1)
        xo = self.outconv(x2)
        return xo


class UNet0(nn.Module):
    def __init__(self, n_channels, n_classes):
        super().__init__()

        self.inconv0 = DoubleConv(n_channels, 64, 64)
        self.inconv1 = DoubleConv(64, 64, 128)
        self.inconv2 = DoubleConv(128, 128, 128)
        self.inconv3 = DoubleConv(128, 128, 256)
        self.inconv4 = DoubleConv(256, 256, 256)

        self.inconv5 = DoubleConv(512, 256, 256)
        self.inconv6 = DoubleConv(384, 192, 128)
        self.inconv7 = DoubleConv(256, 128, 128)
        self.inconv8 = DoubleConv(192, 96, 64)

        self.outconv = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inconv0(x)
        x2 = self.inconv1(x1)
        x3 = self.inconv2(x2)
        x4 = self.inconv3(x3)
        x5 = self.inconv4(x4)
        x = self.inconv5(torch.cat([x4, x5], dim=1))
        x = self.inconv6(torch.cat([x3, x], dim=1))
        x = self.inconv7(torch.cat([x2, x], dim=1))
        x = self.inconv8(torch.cat([x1, x], dim=1))
        x = self.outconv(x)
        return x


class UNet1(nn.Module):
    def __init__(self, n_channels, n_classes):
        super().__init__()

        self.inconv = DoubleConv(n_channels, 64, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 256)
        self.up1 = Up(512, 256)
        self.up2 = Up(384, 128)
        self.up3 = Up(256, 128)
        self.up4 = Up(192, 64)
        self.outconv = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inconv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outconv(x)
        return x


class UNet2(nn.Module):
    def __init__(self, n_channels, n_classes):
        super().__init__()

        self.inconv = DoubleConv(n_channels, 64, 64)
        self.down1 = Down(64, 64)
        self.down2 = Down(64, 64)
        self.down3 = Down(64, 64)
        self.down4 = Down(64, 64)
        self.up1 = Up(128, 64)
        self.up2 = Up(128, 64)
        self.up3 = Up(128, 64)
        self.up4 = Up(128, 64)
        self.outconv = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inconv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outconv(x)
        return x


class UNet3(nn.Module):
    def __init__(self, n_channels, n_classes):
        super().__init__()

        self.inconv = DoubleConv(n_channels, 32, 32)
        self.down1 = Down(32, 32)
        self.down2 = Down(32, 32)
        self.down3 = Down(32, 32)
        self.down4 = Down(32, 32)
        self.up1 = Up(64, 32)
        self.up2 = Up(64, 32)
        self.up3 = Up(64, 32)
        self.up4 = Up(64, 32)
        self.outconv = OutConv(32, n_classes)

    def forward(self, x):
        x1 = self.inconv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outconv(x)
        return x


class AutoEncoder1(nn.Module):
    def __init__(self, n_channels, n_classes):
        super().__init__()

        self.inconv = DoubleConv(n_channels, 64, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 256)
        self.up1 = Up(256, 256)
        self.up2 = Up(256, 128)
        self.up3 = Up(128, 128)
        self.up4 = Up(128, 64)
        self.outconv = OutConv(64, n_classes)

    def forward(self, x):
        x = self.inconv(x)

        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)

        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)

        x = self.outconv(x)
        return x


class AutoEncoder2(nn.Module):
    def __init__(self, n_channels, n_classes):
        super().__init__()

        self.inconv = DoubleConv(n_channels, 64, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 256)
        self.up1 = ConvTrans(256, 256)
        self.up2 = ConvTrans(256, 128)
        self.up3 = ConvTrans(128, 128)
        self.up4 = ConvTrans(128, 64)
        self.outconv = OutConv(64, n_classes)

    def forward(self, x):
        x = self.inconv(x)

        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)

        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)

        x = self.outconv(x)
        return x


class AutoEncoder3(nn.Module):
    def __init__(self, n_channels, n_classes):
        super().__init__()

        self.inconv = DoubleConv(n_channels, 64, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 256)
        self.up1 = ConvTrans(256, 256)
        self.up2 = Up(256, 128)
        self.up3 = ConvTrans(128, 128)
        self.up4 = Up(128, 64)
        self.outconv = OutConv(64, n_classes)

    def forward(self, x):
        x = self.inconv(x)

        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)

        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)

        x = self.outconv(x)
        return x

def get_network(network_id):
    if network_id == "UNet0":
        return UNet0(n_channels=1, n_classes=1)
    elif network_id == "UNet1":
        return UNet1(n_channels=1, n_classes=1)
    elif network_id == "UNet2":
        return UNet2(n_channels=1, n_classes=1)
    elif network_id == "UNet3":
        return UNet3(n_channels=1, n_classes=1)
    if network_id == "CNN1":
        return CNN1(n_channels=1, n_classes=1)
    if network_id == "AE1":
        return AutoEncoder1(n_channels=1, n_classes=1)
    if network_id == "AE2":
        return AutoEncoder2(n_channels=1, n_classes=1)
    if network_id == "AE3":
        return AutoEncoder3(n_channels=1, n_classes=1)
    else:
        raise(IOError("Network ID is not recognized."))

