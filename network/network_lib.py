from math import log2
from network.network_components import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super().__init__()

        self.inconv = DoubleConv(n_channels, 64, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256)
        self.up2 = Up(512, 128)
        self.up3 = Up(256, 64)
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


class UNet64(nn.Module):
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


class UNet64v(nn.Module):
    def __init__(self, n_channels, n_classes, img_res=128):
        super().__init__()
        neck_size = 8
        self.n_down_layers = self.n_up_layers = int(log2(img_res // neck_size))

        self.inconv = DoubleConv(n_channels, 64, 64)

        for i in range(self.n_down_layers):
            self.__dict__["down%s"% (i+1)] = Down(64, 64)

        for i in range(self.n_up_layers):
            self.__dict__["up%s" % (i+1)] = Up(64, 64)

        self.outconv = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inconv(x)
        xdowns = [x1]
        for i in range(self.n_down_layers):
            down_layer = getattr(self, 'down%s'% (i+1))
            xdowns.append(down_layer(xdowns[-1]))

        x = xdowns[-1]
        for i, up_layers in enumerate(self.ups):
            x = up_layers(x, xdowns[self.n_down_layers - i])

        x = self.outconv(x.clone())
        return x

def get_network(network_id):
    if network_id == "UNet":
        return UNet(n_channels=1, n_classes=1)
    elif network_id == "UNet64":
        return UNet64(n_channels=1, n_classes=1)
    elif network_id == "UNet64v":
        return UNet64v(n_channels=1, n_classes=1)

